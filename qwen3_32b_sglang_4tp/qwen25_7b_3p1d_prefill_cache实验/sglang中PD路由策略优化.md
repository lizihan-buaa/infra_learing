# SGLang Gateway HTTP PD Router Route Chat 修改总结

## 1. 前置知识

### HTTP Router 与 PD HTTP Router

在 `sgl-model-gateway` 中，`http` 表示 gateway 和后端 worker 之间使用 HTTP 协议通信。它不是单纯指客户端请求是 HTTP，而是指路由器向后端推理服务转发请求时走 HTTP。

普通 HTTP Router 位于 `sgl-model-gateway/src/routers/http/router.rs`。它用于 regular routing mode，一个请求只选择一个普通 worker，由这个 worker 完整完成 prefill 和 decode，然后把响应返回给客户端。

PD HTTP Router 位于 `sgl-model-gateway/src/routers/http/pd_router.rs`。它用于 Prefill-Decode 分离模式，一个请求会同时涉及两类 worker：

- prefill worker：负责处理输入 prompt，生成 KV cache。
- decode worker：负责后续 token decode，并向客户端返回最终结果。

PD HTTP Router 会选择一组 prefill/decode worker，并给请求注入 `bootstrap_host`、`bootstrap_port`、`bootstrap_room`，让 decode worker 能找到 prefill worker 产生的 KV cache。它还要处理流式响应、非流式响应、重试、熔断、metrics、logprobs 合并等逻辑。

### route_chat 的职责

`route_chat` 是 PD HTTP Router 处理 `/v1/chat/completions` 请求的入口。它本身不直接做 HTTP 双发，而是从 `ChatCompletionRequest` 中提取路由需要的元信息，然后调用 `execute_dual_dispatch`。

它主要准备这些信息：

- `is_stream`：是否流式返回。
- `return_logprob`：是否需要返回 token log probability。
- `request_text`：供 cache-aware 等路由策略使用的请求文本。
- `batch_size`：用于 batch 请求时注入多组 bootstrap 信息。
- `model_id` 和 headers：用于 worker 筛选、转发和策略决策。

### stream 的含义

`stream` 表示流式输出。非流式请求会等待模型完整生成后一次性返回 JSON；流式请求会在模型生成过程中持续返回 SSE chunk，例如：

```text
data: {...}
data: {...}
data: [DONE]
```

在 PD HTTP Router 中，stream 响应主要来自 decode worker。因为流式响应可能中途断开，所以 decode worker 的成功或失败不能只看 HTTP header，需要由 `BreakerTrackedStream` 在流结束、报错或客户端断开时记录。

### logprobs 的含义

`logprobs` 是 token 的对数概率，用来表示模型在每一步生成某个 token 的置信度。PD 模式下 prefill 和 decode 被拆到不同 worker，因此 logprobs 可能分散在两边：

- prefill 侧可能产生输入 token 的 logprobs。
- decode 侧产生输出 token 的 logprobs。

所以 PD HTTP Router 在需要时会合并两边的 logprobs。非流式场景通过 JSON 合并，流式场景则对每个 decode chunk 做必要的 logprobs 合并。

### role 与 messages.first()

Chat 请求中的 `messages` 按 role 区分不同消息来源，代码中涉及的常见 role 包括：

- `system`：系统级指令，通常定义模型整体行为。
- `developer`：开发者级指令。
- `user`：用户输入。
- `assistant`：历史模型回复，可能包含 tool calls。
- `tool`：工具调用结果。
- `function`：旧式 function calling 的函数返回消息。

原来的 `route_chat` 只通过 `body.messages.first()` 取第一条消息，并且只对部分 role 提取文本：

- `system`：提取。
- `developer`：只有纯文本时提取。
- `user`：只有纯文本时提取。
- `assistant`、`tool`、`function`：不提取。
- 多模态 `Parts`：不提取。

提取出的文本会进入 `PDRequestContext.request_text`，再经过 `execute_dual_dispatch -> select_pd_pair -> pick_worker_by_policy_arc -> SelectWorkerInfo.request_text` 传给 cache-aware policy。

### cache-aware policy 如何使用 request_text

cache-aware policy 会维护一个近似 radix tree，用请求文本做前缀匹配，判断哪个 worker 更可能已有相关 KV cache。

当系统负载均衡时，它倾向选择前缀匹配率最高、最可能命中 cache 的 worker；当负载明显不均衡时，它会优先选择负载较低的 worker，同时仍更新 tree 状态。

因此，`request_text` 的质量直接影响 cache-aware 路由效果。文本越接近真实 prompt / 上下文，worker 选择越可能符合实际 KV cache 复用关系。

## 2. SGLang 原本代码面临的问题

### 只看第一条 message，路由文本代表性不足

原逻辑使用 `body.messages.first()`。这对简单请求可以工作，但对真实 Chat Completion 请求并不稳健。

很多业务请求的第一条消息是固定 system prompt，例如：

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Explain PD routing in SGLang"}
]
```

原逻辑提取到的是：

```text
You are a helpful assistant.
```

而不是用户真正的问题。这样大量不同请求会因为相同 system prompt 被视为相似，cache-aware 的前缀匹配依据会变粗，worker 亲和性不够准确。

### 多轮对话和长上下文场景下 cache affinity 不充分

Chat 请求常常包含完整历史上下文：

```text
system + 历史 user/assistant + 当前 user
```

KV cache 的复用通常与完整上下文前缀相关，而不是只与第一条消息相关。只看第一条 message 会忽略历史对话和当前用户问题，导致 cache-aware policy 无法充分利用实际上下文相似性。

这在以下场景中尤其明显：

- 多轮聊天。
- 长上下文续写。
- RAG 请求，前面有大量检索上下文。
- agent/tool calling 后继续对话。
- 固定 system prompt + 多种用户问题的业务助手。

### 对 role 和内容类型过于敏感

原逻辑只处理 `system`、纯文本 `developer`、纯文本 `user`。如果第一条是 `assistant`、`tool`、`function`，或者第一条是多模态 `Parts`，`request_text` 就会是 `None`。

这意味着 cache-aware 在这些请求上拿不到有效文本，只能退化成空文本或其他 fallback 逻辑。

### PD HTTP Router 与普通 HTTP Router 行为不一致

普通 HTTP Router 已经通过 `typed_req.extract_text_for_routing()` 提取路由文本，而 PD HTTP Router 对 chat 请求手写了一套 first-message-only 逻辑。

这会造成同一个 Chat 请求在 regular HTTP 模式和 PD HTTP 模式下使用不同的路由文本。结果是：

- 行为不一致。
- 后续维护成本更高。
- 协议层文本提取逻辑优化后，PD HTTP Router 不能自动受益。

### cache-aware 的收益被削弱

cache-aware policy 的目标是通过 prompt / 上下文相似性提高 KV cache 命中率。如果传入的 `request_text` 只是一段固定 system prompt，或者根本为空，就会削弱它的核心价值。

最终表现可能是：

- 相似请求没有稳定落到更合适的 worker。
- 不相似请求因为共享 system prompt 被错误聚合。
- prefill/decode 两侧的 cache-aware 选择依据不够精确。
- 长上下文场景下 KV cache 复用率不如预期。

## 3. 解决方法

### 修改 route_chat 的 request_text 提取方式

将原来的 first-message-only 逻辑：

```rust
let request_text = if self.policies_need_request_text() {
    body.messages.first().and_then(|msg| match msg {
        ChatMessage::User { content, .. } => match content {
            MessageContent::Text(text) => Some(text.clone()),
            MessageContent::Parts(_) => None,
        },
        ChatMessage::Developer { content, .. } => match content {
            MessageContent::Text(text) => Some(text.clone()),
            MessageContent::Parts(_) => None,
        },
        ChatMessage::System { content, .. } => Some(content.to_simple_string()),
        _ => None,
    })
} else {
    None
};
```

替换为：

```rust
let request_text = if self.policies_need_request_text() {
    let text = body.extract_text_for_routing();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
} else {
    None
};
```

### 这个修改带来的优化

第一，cache-aware 使用的文本更接近真实请求上下文。新的 `extract_text_for_routing()` 会按协议层统一逻辑提取可用于路由的文本，而不是只取第一条 message。这样 cache-aware radix tree 的前缀匹配更接近实际 KV cache 内容。

第二，多轮对话和长上下文场景下更容易获得正确的 cache affinity。请求之间如果共享完整上下文前缀，就更可能被路由到已有相关 cache 的 worker。

第三，PD HTTP Router 和普通 HTTP Router 行为对齐。普通 HTTP Router 已经在 `route_typed_request` 中使用 `extract_text_for_routing()`，PD HTTP Router 改为同样方式后，同类请求在 regular 和 PD 模式下会使用一致的路由文本。

第四，维护性更好。文本提取逻辑集中在协议层方法中，PD Router 不需要自己维护一份 role 匹配逻辑。后续如果协议层支持更多消息格式或优化提取方式，PD Router 能自然复用。

第五，开销仍然受控。代码保留了：

```rust
self.policies_need_request_text()
```

只有当前 prefill 或 decode policy 需要文本时才调用 `extract_text_for_routing()`。如果使用 random、round-robin、power-of-two 等不依赖文本的策略，就不会遍历 messages 和构造拼接字符串。

第六，空文本语义更清楚。新逻辑把空字符串转成 `None`，表示没有可用于路由的有效文本，而不是继续传递 `Some("")`。

### 更适合的场景

这个修改尤其适合使用 `cache_aware` 策略的 PD HTTP 部署，特别是以下场景：

- 请求有固定 system prompt，但 user 内容变化较大。
- 多轮聊天请求较多。
- RAG 请求中包含较长检索上下文。
- 同一用户或同一业务会话经常复用相似上下文。
- 长上下文模型服务，希望提高 KV cache 复用率。
- regular HTTP 和 PD HTTP 模式需要尽量保持一致路由行为。

### 需要注意的边界

这个修改会在 text-aware policy 下遍历更多 message，并构造更长的 routing text。相比只 clone 第一条消息，CPU 和内存开销会略高。但它只在策略确实需要文本时发生，且对 cache-aware 的路由准确性收益通常更大。

另外，HTTP PD 路径中传给 `SelectWorkerInfo` 的 `tokens` 仍然是 `None`。因此这个修改主要优化依赖 `request_text` 的策略，例如 `cache_aware` 和文本长度相关策略。对于真正依赖 token 序列的 prefix-hash，需要额外的 tokenized input 支持才能完整发挥作用。

整体来说，这个修改把 PD HTTP Chat 路由从“基于第一条消息的粗粒度文本亲和”提升为“基于完整可路由文本的上下文亲和”，更适合真实多轮 Chat、长上下文和 RAG 负载。
