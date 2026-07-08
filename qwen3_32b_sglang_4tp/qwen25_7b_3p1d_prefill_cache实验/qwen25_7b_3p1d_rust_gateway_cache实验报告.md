# Qwen2.5-7B 3P1D Rust Gateway 多轮前缀缓存实验报告

## 本轮定位到的问题

1. 之前 benchmark 把 SSE 流里的 `error` chunk 当成了正常响应。SGLang PD transfer 失败时，HTTP 仍可能返回 200，但流内第一条 data 是 `Decode transfer failed ...`，后面还会带 usage chunk。脚本现在只要看到 `error` 对象就把请求记为失败，不再统计进 cache hit rate。
2. 36 会话三轮请求会在第三轮触发大量 Mooncake KV transfer 失败。重启 worker 并增加 `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=true` 后，短请求和前两轮 benchmark 可以稳定完成；正式对比因此固定为 `--max-turns 2`。
3. 12 会话加高亲和阈值会让原版和优化版都集中到同一 prefill worker，复用率都会接近 99.7%，无法体现路由策略差异。最终使用默认负载阈值 `balance_abs_threshold=4`、`balance_rel_threshold=1.2`，并把会话数提高到 36。

## Benchmark 修正

脚本：`pd_rust_router_same_context_benchmark.py`

关键修正：

1. 每个会话的稳定长上下文以会话唯一前缀开头，后续轮只替换末尾问题，避免跨会话公共前缀掩盖路由差异。
2. `max_tokens` 从 1 改为 8，避免 completion 为 0 时 TTFT 口径失真。
3. 增加 `--max-turns`，正式对比只跑 `seed_turn1` 和 `warm_turn2`。
4. 流式响应中出现 `error` chunk 时，该请求计为失败。

## 实验配置

- 模型：`/DaTa/lizihan/models/Qwen2.5-7B-Instruct`
- 拓扑：3P1D，GPU0/1/2 为 prefill，GPU3 为 decode
- SGLang worker：`--enable-cache-report --tp-size 1 --context-length 4096 --attention-backend torch_native --disable-cuda-graph`
- PD transfer：`--disaggregation-transfer-backend mooncake_tcp`
- Gateway policy：`--prefill-policy cache_aware --decode-policy round_robin`
- 对比阈值：`--cache-threshold 0.3 --balance-abs-threshold 4 --balance-rel-threshold 1.2`
- 请求：36 个会话，每个会话 2 轮；seed 并发 36，warm 并发 12

## 最终结果

| 策略 | 请求数 | 失败数 | 总 cache hit rate | warm2 cache hit rate | 总平均 TTFT(s) | warm2 平均 TTFT(s) |
|---|---:|---:|---:|---:|---:|---:|
| original_first_message | 72 | 0 | 0.0823 | 0.1530 | 4.888 | 3.919 |
| optimized_full_conversation | 72 | 0 | 0.0991 | 0.1866 | 4.572 | 4.092 |

## 结论

1. 优化版体现出更高的 cache 复用率：warm2 从 15.30% 提升到 18.66%，提升 3.36 个百分点，约 22% 相对提升。
2. 差异来自路由文本质量：原版只用第一条 message，当前请求的第一条 system message 对所有会话相同；优化版使用完整 conversation text，能把同一会话后续请求更稳定地路由回持有该长上下文 KV cache 的 prefill worker。
3. TTFT 不能单独等价为 cache 命中率。当前 3P1D 下 TTFT 同时受 prefill 队列、Mooncake transfer、decode 排队和流式首包影响；因此报告同时保留 `cached_tokens/prompt_tokens` 和 TTFT。
4. 大负载三轮会暴露 Mooncake transfer 稳定性问题，后续如果继续加压，应先解决 `remote mooncake session ... is not alive`，否则会把路由效果和传输失败混在一起。

## 结果文件

- 优化版结果：`/tmp/sglang_pd_final_results/optimized_36_2turn`
- 原版结果：`/tmp/sglang_pd_final_results/original_36_2turn`
- 实验脚本：`pd_rust_router_same_context_benchmark.py`
