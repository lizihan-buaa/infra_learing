# Qwen2.5-7B-Instruct 3P1D Prefill Cache 命中实验报告

## 1. 实验目标

在当前服务器使用前四张 RTX 5090 GPU 运行 SGLang PD 分离推理：3 个 prefill worker + 1 个 decode worker。目标是只关注 prefill 阶段，比较多轮对话在优化前后路由策略下的 prefix/KV cache 命中率变化。

本次实验没有使用容器镜像。模型权重使用：`/DaTa/lizihan/models/Qwen2.5-7B-Instruct`。环境使用：`/home/lizihan/sglang_cp_env`。

## 2. 环境与启动配置

- SGLang：`0.5.13.post1`
- PyTorch：`2.11.0+cu128`
- GPU：使用 GPU 0、1、2、3，每张卡 32GB
- 模型：Qwen2.5-7B-Instruct
- 并行方式：PD 分离，prefill worker 单卡 TP=1，decode worker 单卡 TP=1
- transfer backend：`mooncake_tcp`，即 SGLang PD KV 传输走 Mooncake TCP 后端
- attention backend：`torch_native`
- cache report：开启 `--enable-cache-report`

重要修正：必须清空代理变量并设置本地 NO_PROXY，否则 decode 访问 `127.0.0.1:bootstrap_port/route` 会被系统代理转发到 squid，导致 403 和 PD handshake 失败。

## 3. 关键启动命令

### Prefill worker 0, GPU0

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
CUDA_VISIBLE_DEVICES=0 /home/lizihan/sglang_cp_env/bin/python -m sglang.launch_server \
  --model-path /DaTa/lizihan/models/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 32100 --tp 1 \
  --mem-fraction-static 0.58 --context-length 4096 \
  --enable-cache-report \
  --attention-backend torch_native --sampling-backend pytorch \
  --disable-cuda-graph --disable-piecewise-cuda-graph --skip-server-warmup \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 9110
```

Prefill worker 1/2 分别使用 GPU1/GPU2，HTTP 端口 `32101/32102`，bootstrap 端口 `9111/9112`。

### Decode worker, GPU3

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
CUDA_VISIBLE_DEVICES=3 /home/lizihan/sglang_cp_env/bin/python -m sglang.launch_server \
  --model-path /DaTa/lizihan/models/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 32110 --tp 1 \
  --mem-fraction-static 0.58 --context-length 4096 \
  --enable-cache-report \
  --attention-backend torch_native --sampling-backend pytorch \
  --disable-cuda-graph --disable-piecewise-cuda-graph --skip-server-warmup \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake_tcp
```

## 4. Benchmark 设计

由于当前环境没有可运行的 Rust `sgl-model-gateway` 二进制，也没有 `cargo` 构建入口；Python `sglang_router` 的 Rust binding 也未安装，所以没有直接运行官方 cache-aware router。为完成 prefill cache 命中验证，我写了直接 PD benchmark：每个请求同时发给一个 prefill worker 和固定 decode worker，并显式传入：

- `bootstrap_host`
- `bootstrap_port`
- `bootstrap_room`

这与 SGLang MiniLB 的 PD 请求字段一致，只是路由选择由 benchmark 控制。

测试两种策略：

1. `random`：每轮请求随机选择 prefill worker，模拟优化前路由不关注多轮对话上下文归属。
2. `sticky`：同一个 conversation id 固定路由到同一个 prefill worker，模拟针对多轮对话 cache 命中的会话粘性优化。

每种策略使用 5 个会话，每个会话 4 轮，总计 20 个 PD 请求。每次请求 `max_tokens=1`，只让 decode 输出 1 个 token，尽量把观察重点放在 prefill cache 命中。

Benchmark 脚本：`/home/lizihan/docs/pd_3p1d_direct_benchmark.py`

结果日志：`/tmp/sglang_pd_3p1d_logs/direct_benchmark_random_vs_sticky.jsonl`

## 5. 实验结果

| 策略 | 请求数 | 成功数 | prompt tokens | cached tokens | cache hit rate | 平均延迟 | P50 延迟 | 最大延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 20 | 20 | 2520 | 963 | 38.21% | 44.28 ms | 38.86 ms | 120.55 ms |
| sticky | 20 | 20 | 2520 | 1685 | 66.87% | 37.87 ms | 37.64 ms | 42.80 ms |

结论：

1. 3P1D 链路跑通。最终 40 个 PD 请求全部成功，没有再出现 `Access Denied`、`Decode handshake failed` 或 HTTP 500。
2. 会话粘性路由明显提高 prefill cache 命中率：从 `38.21%` 提升到 `66.87%`，提升约 `28.65` 个百分点。
3. sticky 策略降低了尾延迟：最大延迟从 `120.55 ms` 降到 `42.80 ms`。平均延迟也从 `44.28 ms` 降到 `37.87 ms`。
4. random 策略下，同一会话的后续轮次可能落到不同 prefill worker，只能命中公共 system prompt 或短公共前缀；sticky 策略下，同一会话历史持续落到同一 worker，后续轮次能命中完整历史前缀。

## 6. 观察到的问题与处理

1. 第一次实验失败原因：代理环境变量污染。decode 内部访问 `http://127.0.0.1:9111/route?...` 被转发到 squid，返回 403。解决方式是在所有 worker 和 benchmark 命令中清空 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`，并设置 `NO_PROXY=127.0.0.1,localhost`。
2. 日志中有 `deep_gemm/_C.so libcudart.so.13` 的导入错误，但这是 SGLang 扫描 MoE/DeepSeek/GLM 等模型模块时的可忽略错误。本次 Qwen2.5-7B dense 模型没有使用这些模块。
3. 当前机器没有可直接使用的 Rust gateway，因此本报告中的“优化前/后”是通过 benchmark 控制 prefill 路由策略实现的，不是官方 Rust `cache_aware` router 的端到端测试。要复现真正的 router 优化，需要先安装或构建 `sgl-model-gateway` 的 Rust/Python binding。

## 7. 后续建议

1. 安装可运行的 `sgl-model-gateway` Rust binary 或 Python `sglang_router` binding 后，再用官方 `--prefill-policy cache_aware` 跑同一套 benchmark。
2. 扩大请求规模，例如 50 到 200 个会话，每个会话 6 到 10 轮，观察 cache 命中率和尾延迟是否稳定。
3. 把随机路由、会话 sticky、真实 cache-aware 三种策略放在同一组冷启动条件下对比，避免已有 cache 对结果产生偏置。
