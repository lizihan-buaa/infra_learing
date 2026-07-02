# Qwen3-32B 4TP Prefill Concurrency Sweep

实验设置：

- Model: `/DaTa/lizihan/weight/Qwen3-32B`
- Runtime: SGLang, BF16, TP=4, GPUs 0,1,2,3
- Backend: `--attention-backend triton --sampling-backend pytorch`
- Input length: 4096 tokens fixed
- Output length: 1 token
- Dataset: `random-ids`
- `--random-range-ratio 1`
- Requests per point: 50
- Warmup requests: 2
- Radix cache disabled and cache flushed before each run

| Max concurrency | Benchmark duration (s) | Req/s | Input tok/s | Mean TTFT (ms) | Median TTFT (ms) | P90 E2E (ms) | P99 TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.41 | 1.50 | 6130.65 | 654.15 | 660.84 | 746.30 | 837.59 |
| 2 | 32.89 | 1.52 | 6226.51 | 1282.48 | 1291.72 | 1482.93 | 1568.79 |
| 4 | 32.14 | 1.56 | 6371.48 | 2460.57 | 2461.35 | 2791.51 | 2845.10 |
| 8 | 32.15 | 1.55 | 6369.25 | 4729.04 | 5122.00 | 5486.49 | 5552.56 |
| 10 | 32.14 | 1.56 | 6372.27 | 5782.22 | 6386.75 | 6752.29 | 6813.11 |
| 16 | 32.14 | 1.56 | 6372.68 | 8628.42 | 10003.98 | 10461.92 | 10790.87 |
| 24 | 32.15 | 1.56 | 6371.11 | 11684.93 | 15053.31 | 15539.30 | 15560.62 |
| 32 | 32.16 | 1.55 | 6368.36 | 13948.17 | 15882.86 | 20628.89 | 20732.66 |

结论：

- 严格按最高 input throughput 看，并发 16 最高，为 6372.68 tok/s。
- 但并发 4、10、16 的吞吐差距小于 0.03%，属于噪声级别。
- 并发 4 是更合理的最优点：吞吐已经进入平台期，同时 TTFT 明显低于 8/10/16/24/32。
- 如果目标是最低延迟，选择并发 1。
- 如果目标是吞吐优先且能接受约 2.5s 平均 TTFT，选择并发 4。
- 不建议为了吞吐使用 8 以上并发，因为吞吐没有明显提升，TTFT 和尾延迟快速恶化。
