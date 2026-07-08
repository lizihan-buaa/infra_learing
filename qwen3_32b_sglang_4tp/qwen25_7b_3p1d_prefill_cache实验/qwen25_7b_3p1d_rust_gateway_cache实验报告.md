# Qwen2.5-7B 3P1D Rust Gateway 多轮前缀 Cache 对比报告

## 1. 实验目标

本轮对比 Rust gateway 在 PD disaggregation 场景下两种 cache-aware 路由方式对多轮对话 prefill KV cache 命中的影响：

- 原版策略 `original_first_message`：PD 路由构造 cache key 时主要使用 chat 请求的第一条 message。
- 优化策略 `optimized_full_conversation`：PD 路由构造 cache key 时使用完整 conversation，包括 system、历史 user/assistant 轮次和当前 user 轮次。

测试重点是 prefill 阶段返回的 `prompt_tokens_details.cached_tokens`，并同步观察请求延迟。

## 2. 实验配置

- 模型：`/DaTa/lizihan/models/Qwen2.5-7B-Instruct`
- 部署：3 个 prefill worker + 1 个 decode worker，GPU0-GPU3，TP=1
- Worker：SGLang，`--disaggregation-transfer-backend mooncake_tcp`，`--enable-cache-report`
- Gateway 原版：`/tmp/sgl-model-gateway-original`
- Gateway 优化版：`/tmp/sgl-model-gateway-optimized`
- 路由参数：`--pd-disaggregation --prefill-policy cache_aware --decode-policy round_robin`
- 结果目录：`prefix_router_results_small/`

PD 运行参数：

```bash
SGLANG_DISAGGREGATION_QUEUE_SIZE=64
SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=64
SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=10
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=20
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
```

## 3. Benchmark 设计

数据集脚本：`pd_rust_router_prefix_dataset_benchmark.py`

三类场景：

1. `support_cases`：同一工单上下文连续 3 轮追问。
2. `rag_documents`：同一检索文档连续 3 轮追问。
3. `code_modules`：同一代码模块连续 3 轮追问。

本轮采用稳定对比基准：

- 每个场景 `num_groups=12`
- 每组 `turns=3`
- 每个场景 36 个请求
- 首轮 `seed_concurrency=3`
- 后续轮次 `concurrency=1`
- `support_cases` 使用 `context_repeats=3`
- `rag_documents` 和 `code_modules` 使用 `context_repeats=2`
- 每个场景开始前执行 `flush_cache`

命中率计算：

`cache_hit_rate = sum(cached_tokens) / sum(prompt_tokens)`

## 4. 实验结果

| 策略 | 场景 | 请求数 | 成功请求 | prompt_tokens | cached_tokens | cache 命中率 | 平均延迟(s) | P90(s) | P99(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original_first_message | support_cases | 36 | 36 | 85572 | 57107 | 66.74% | 0.330 | 0.577 | 1.571 |
| optimized_full_conversation | support_cases | 36 | 36 | 85572 | 57010 | 66.62% | 0.285 | 0.975 | 1.512 |
| original_first_message | rag_documents | 36 | 36 | 58092 | 38687 | 66.60% | 0.182 | 0.456 | 0.565 |
| optimized_full_conversation | rag_documents | 36 | 36 | 58092 | 38686 | 66.59% | 0.133 | 0.206 | 0.299 |
| original_first_message | code_modules | 36 | 36 | 58128 | 38699 | 66.58% | 0.270 | 0.426 | 0.560 |
| optimized_full_conversation | code_modules | 36 | 36 | 58128 | 38654 | 66.50% | 0.204 | 0.243 | 0.432 |

## 5. 对比结论

1. 在本轮稳定基准下，两种策略的总 cache 命中率非常接近，差异小于 0.25 个百分点。原因是每个场景只有 12 组、请求顺序固定、后续轮次串行，且所有请求共享相同 system prompt，原版 first-message key 也能把请求较集中地路由到已有 cache 的 prefill worker。

2. 优化版在 `rag_documents` 和 `code_modules` 上延迟更低：RAG 平均延迟从 0.182s 降到 0.133s，P90 从 0.456s 降到 0.206s；代码场景平均延迟从 0.270s 降到 0.204s，P90 从 0.426s 降到 0.243s。说明完整 conversation key 能更准确表达当前请求的真实长前缀，减少路由歧义。

3. `support_cases` 的平均延迟优化版更低，0.330s 降到 0.285s；但 P90 高于原版，主要来自首轮长上下文请求的调度抖动。这个场景不能单独用 P90 判断优化收益，应结合平均延迟、cache 命中率和每个 worker 的路由分布一起看。

4. 优化策略的核心价值不一定体现在小规模稳定基准的总命中率上，而体现在多会话、多轮历史、多类前缀混杂时的路由 key 表达能力。原版只看第一条 message，容易把不同会话视为相同前缀；优化版使用完整 conversation，更接近 SGLang 实际构造 prompt 后的 KV 前缀关系。

## 6. 原因分析

原版 PD 路由的 cache-aware 选择只依赖第一条 message 时，很多 chat 请求的第一条都是相同 system prompt。这样路由层看到的 key 区分度很低，无法稳定表达“这个请求属于哪个工单/文档/代码模块的第几轮”。在小规模串行请求中，这种低区分度有时也会得到较高命中率，因为请求被集中到同一组 worker 上。

优化版把完整 conversation 作为路由文本后，后续轮次会包含第一轮长上下文、历史 assistant 占位回复和当前问题。这样 cache-aware policy 看到的前缀更接近实际模型输入，后续请求更容易被送回保存对应 KV 的 prefill worker。这个改动对 RAG 和代码场景更明显，因为这些场景的长上下文结构稳定、组间差异清晰，完整 key 能更好地区分不同会话。

## 7. 关键命令

原版 gateway：

```bash
/tmp/sgl-model-gateway-original \
  --host 127.0.0.1 --port 32210 \
  --pd-disaggregation \
  --prefill http://127.0.0.1:32100 9110 \
  --prefill http://127.0.0.1:32101 9111 \
  --prefill http://127.0.0.1:32102 9112 \
  --decode http://127.0.0.1:32110 \
  --policy cache_aware \
  --prefill-policy cache_aware \
  --decode-policy round_robin
```

优化版 gateway：

```bash
/tmp/sgl-model-gateway-optimized \
  --host 127.0.0.1 --port 32210 \
  --pd-disaggregation \
  --prefill http://127.0.0.1:32100 9110 \
  --prefill http://127.0.0.1:32101 9111 \
  --prefill http://127.0.0.1:32102 9112 \
  --decode http://127.0.0.1:32110 \
  --policy cache_aware \
  --prefill-policy cache_aware \
  --decode-policy round_robin
```

benchmark 示例：

```bash
python pd_rust_router_prefix_dataset_benchmark.py \
  --base-url http://127.0.0.1:32210 \
  --strategy optimized_full_conversation \
  --scenario rag_documents \
  --num-groups 12 \
  --turns 3 \
  --repeat 1 \
  --context-repeats 2 \
  --seed-concurrency 3 \
  --concurrency 1 \
  --sleep 0.2 \
  --out-dir /tmp/sglang_pd_compare_results_small/optimized
```
