# Qwen3-32B Dense BF16 SGLang 4TP 实验汇总

## 1. 实验目标与结论概览

本实验在当前服务器上使用 4 张 RTX 5090 对 Qwen3-32B dense BF16 进行 SGLang 分布式推理、prefill benchmark 和并行策略对比，重点关注：

- 4TP 服务能否稳定启动；
- 4096 token prefill 在不同并发下的吞吐和 TTFT；
- 40000 token 长上下文 prefill 下，纯 4TP 与 attention TP+CP 的性能差异；
- piecewise CUDA graph 和 KV capacity 对结果的影响；
- 当前环境中遇到的普适性问题及解决方法。

主要结论：

- 当前 4x RTX 5090 PCIe-only 环境下，Qwen3-32B dense BF16 可以稳定以 SGLang 4TP 方式运行。
- 对 4096 token prefill，吞吐在并发 4 左右进入平台期；继续提高并发几乎不提升吞吐，但显著增加 TTFT 和尾延迟。
- 对 40000 token 单并发 prefill，纯 4TP 明显快于 SGLang 普通 attention TP+CP。
- 纯 4TP 禁用 piecewise CUDA graph 后性能基本不变，因此 TP+CP 变慢的主因不是 piecewise CUDA graph 不公平。
- 把纯 4TP 的 KV capacity 限制到与 TP+CP 接近后，单并发性能仍基本不变；KV capacity 对单条 40k 请求不是主瓶颈，但会限制多并发长上下文请求。

## 2. 硬件与软件环境

- 服务器 GPU：8 x NVIDIA GeForce RTX 5090，每张约 32 GB 显存。
- 本次实验 GPU：`0,1,2,3`。
- GPU 互联：PCIe-only，未观察到 NVLink。
- 模型：`/DaTa/lizihan/weight/Qwen3-32B`。
- 模型类型：Qwen3-32B dense。
- 权重精度：BF16。
- 推理框架：SGLang `0.5.13.post1`。
- Python 环境：`/home/lizihan/sglang_cp_env/bin/python`。
- Torch：`2.11.0+cu128`。
- CUDA runtime：12.8。
- Nsight Systems：`/usr/local/cuda-12.8/bin/nsys`。
- 可跑通 backend：
  - `--attention-backend triton`
  - `--sampling-backend pytorch`

默认 FlashInfer attention 未采用。当前 RTX 5090 / SM120 + CUDA 12.8 组合下，FlashInfer 相关路径可能报：

```text
SM 12.x requires CUDA >= 12.9
FlashInfer requires GPUs with sm75 or higher
```

因此统一使用 Triton attention 和 PyTorch sampling。

## 3. 服务启动与停止

### 3.1 纯 4TP 启动命令

启动服务时必须清理代理环境，否则服务端或 benchmark 客户端访问 `127.0.0.1` 可能被代理劫持。

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
CC=/home/lizihan/bin/gcc-python312 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/lizihan/sglang_cp_env/bin/python -m sglang.launch_server \
  --model-path /DaTa/lizihan/weight/Qwen3-32B \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.88 \
  --context-length 32768 \
  --disable-radix-cache \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-custom-all-reduce \
  --skip-server-warmup \
  --log-level info
```

服务 ready 的标志：

```text
The server is fired up and ready to roll!
Uvicorn running on http://0.0.0.0:30000
```

本机验证：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
curl -sS http://127.0.0.1:30000/v1/models
```

返回中应包含：

```text
/DaTa/lizihan/weight/Qwen3-32B
```

### 3.2 40k 上下文纯 4TP 启动命令

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
CC=/home/lizihan/bin/gcc-python312 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/lizihan/sglang_cp_env/bin/python -m sglang.launch_server \
  --model-path /DaTa/lizihan/weight/Qwen3-32B \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.88 \
  --context-length 40960 \
  --disable-radix-cache \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-custom-all-reduce \
  --skip-server-warmup \
  --log-level info
```

### 3.3 Attention TP+CP 启动命令

Qwen3 dense 可使用 SGLang 普通 prefill context parallel，而不是 DSA/NSA 专用路径。

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
CC=/home/lizihan/bin/gcc-python312 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/lizihan/sglang_cp_env/bin/python -m sglang.launch_server \
  --model-path /DaTa/lizihan/weight/Qwen3-32B \
  --tensor-parallel-size 4 \
  --attention-context-parallel-size 2 \
  --enable-prefill-context-parallel \
  --prefill-cp-mode in-seq-split \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.88 \
  --context-length 40960 \
  --disable-radix-cache \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-custom-all-reduce \
  --skip-server-warmup \
  --log-level info
```

该配置下：

```text
world_size = 4
attn_cp_size = 2
effective_attention_tp_size = 4 / 2 = 2
```

也就是 attention prefill 内部为 `2TP x 2CP`。不要对 Qwen3 dense 使用：

```bash
--enable-dsa-prefill-context-parallel
--dsa-prefill-cp-mode round-robin-split
```

该路径面向 GLM/DeepSeek 等 DSA/NSA 稀疏 attention 模型，不适合 Qwen3 dense。

### 3.4 停止服务

查询进程：

```bash
pgrep -af 'sglang.launch_server|sglang serve|sglang::scheduler|sglang::detokenizer|/home/lizihan/sglang_cp_env/bin/python'
```

如果服务在当前终端前台运行，使用：

```text
Ctrl-C
```

如果服务在后台运行，按主进程 PID 终止：

```bash
kill <sglang_main_pid>
```

若仍有残留子进程，可继续终止：

```bash
pkill -f 'sglang.launch_server'
pkill -f '/home/lizihan/sglang_cp_env/bin/python.*sglang'
pkill -f 'sglang::scheduler'
pkill -f 'sglang::detokenizer'
```

确认 GPU 0-3 已释放：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

正常停止后 GPU 0-3 应回到约 `4 MiB`。

## 4. Benchmark 方法

### 4.1 4096 token prefill benchmark

使用 `random-ids`，避免联网下载 ShareGPT。

重要参数：

- `--random-range-ratio 1`：保证输入长度固定。
- `--random-output-len 1`：近似只测 prefill。
- `--flush-cache`：每轮前清 cache。

单并发示例：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
/home/lizihan/sglang_cp_env/bin/python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random-ids \
  --random-input-len 4096 \
  --random-output-len 1 \
  --random-range-ratio 1 \
  --num-prompts 10 \
  --request-rate inf \
  --max-concurrency 1 \
  --warmup-requests 1 \
  --flush-cache
```

10 并发示例：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
/home/lizihan/sglang_cp_env/bin/python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random-ids \
  --random-input-len 4096 \
  --random-output-len 1 \
  --random-range-ratio 1 \
  --num-prompts 10 \
  --request-rate inf \
  --max-concurrency 10 \
  --warmup-requests 1 \
  --flush-cache
```

### 4.2 40000 token prefill benchmark

40k 测试必须使用 `--tokenize-prompt`，否则 `random-ids` 生成的字符串经过 tokenizer 后会膨胀到约 41k-44k token，超过 Qwen3-32B 的 `40960` context limit，导致无效结果。

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
/home/lizihan/sglang_cp_env/bin/python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random-ids \
  --random-input-len 40000 \
  --random-output-len 1 \
  --random-range-ratio 1 \
  --num-prompts 10 \
  --request-rate inf \
  --max-concurrency 1 \
  --flush-cache \
  --tokenize-prompt
```

## 5. 4096 Token Prefill 并发结果

### 5.1 单并发与 10 并发初测

| 并发 | Total input tokens | Input token throughput | Mean TTFT | P99 TTFT |
| --- | ---: | ---: | ---: | ---: |
| 1 | 40960 | 6026.85 tok/s | 677.36 ms | 817.97 ms |
| 10 | 40960 | 6347.20 tok/s | 3807.79 ms | 6405.29 ms |

说明：

- TTFT 是单个请求从发出到首 token 返回的耗时，不是所有请求 prefill 完成的总耗时。
- 10 并发下每个请求的 TTFT 会显著上升，因为请求在服务端合批和排队。

### 5.2 并发 sweep

实验设置：

- 输入长度：4096 tokens fixed。
- 输出长度：1 token。
- 请求数：每个并发点 50 条。
- Warmup requests：2。
- Radix cache disabled，并在每轮前 flush cache。

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

- 严格按最高 input throughput 看，并发 16 最高，为 `6372.68 tok/s`。
- 但并发 4、10、16 的吞吐差距小于 `0.03%`，属于噪声级别。
- 并发 4 是更合理的最优点：吞吐已经进入平台期，同时 TTFT 明显低于 8/10/16/24/32。
- 如果目标是最低延迟，选择并发 1。
- 如果目标是吞吐优先且能接受约 2.5s 平均 TTFT，选择并发 4。
- 不建议为了吞吐使用 8 以上并发，因为吞吐没有明显提升，TTFT 和尾延迟快速恶化。

## 6. 40000 Token Prefill: 纯 4TP 与 Attention TP+CP 对比

### 6.1 实验设置

- Context length：40960。
- 输入长度：40000 token。
- 输出长度：1 token。
- 请求数：10。
- 并发：1。
- Dataset：`random-ids`。
- 使用 `--tokenize-prompt`。
- `--random-range-ratio 1`。
- `--flush-cache`。

### 6.2 对比结果

| Mode | Load memory / GPU | KV tokens | CUDA graph | Duration | Input tok/s | Mean E2E | Mean TTFT |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Pure 4TP | 15.50 GB | 187069 | piecewise enabled | 107.15 s | 3733.14 | 10710.43 ms | 5353.43 ms |
| Pure 4TP, piecewise disabled | 15.50 GB | 187069 | piecewise disabled | 106.70 s | 3748.68 | 10666.13 ms | 5332.14 ms |
| Attention 2TP x 2CP | 18.25 GB | 70950 | piecewise disabled | 147.67 s | 2708.69 | 14762.58 ms | 7383.16 ms |

相对结果：

- TP+CP input throughput 比纯 4TP 低约 `27.4%`。
- 与同样禁用 piecewise CUDA graph 的纯 4TP 相比，TP+CP input throughput 低约 `27.7%`。
- TP+CP mean E2E latency 比纯 4TP 高约 `37.8%`；比禁用 piecewise 的纯 4TP 高约 `38.4%`。
- TP+CP 使用更多 weight memory，并留下更少 KV capacity。

结论：

- 当前 4x RTX 5090 PCIe-only 环境下，Qwen3-32B dense 40k prefill 使用纯 4TP 明显优于普通 attention TP+CP。
- 纯 4TP 禁用 piecewise CUDA graph 后性能几乎不变，因此性能差距不是由 piecewise CUDA graph 不公平导致。
- TP+CP 变慢的主要原因更可能是：
  - `--attention-context-parallel-size 2` 让 attention 变成有效 `2TP x 2CP`，attention TP 从 4 降到 2；
  - CP 增加 PCIe 通信；
  - CP 模式改变 cache 和显存布局；
  - 每卡 loaded memory 从 15.50 GB 增至 18.25 GB，KV capacity 从 187k 降至 71k。

## 7. KV Capacity 控制实验

为了判断纯 4TP 与 TP+CP 的差距是否来自 KV capacity，将纯 4TP 的 `max_total_tokens` 手动限制到与 TP+CP 接近：

```bash
--max-total-tokens 70950
--disable-piecewise-cuda-graph
```

启动后日志确认：

```text
max_total_num_tokens=70950
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 70950
```

### 7.1 单并发结果

| Mode | KV tokens | Max concurrency | Duration | Input tok/s | Mean E2E | Mean TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure 4TP, piecewise disabled | 187069 | 1 | 106.70 s | 3748.68 | 10666.13 ms | 5332.14 ms |
| Pure 4TP, piecewise disabled, KV limited | 70950 | 1 | 106.55 s | 3754.26 | 10650.06 ms | 5324.23 ms |
| Attention 2TP x 2CP | 70950 | 1 | 147.67 s | 2708.69 | 14762.58 ms | 7383.16 ms |

结论：

- 对单条 40k 请求，只要 KV capacity 能容纳请求，capacity 大小不是主要性能瓶颈。
- 纯 4TP 即使限制到 70950 KV tokens，单并发 40k 性能仍基本不变。
- 因此 TP+CP 慢主要不是因为 KV capacity 小，而是 attention 并行组织和通信路径不同。

### 7.2 4 并发观察

纯 4TP 限制到 70950 KV tokens 后，4 条 40k 请求总需求约 160k token，明显超过可用 KV capacity。

实际观察：

- 4 并发测试被手动中止，未得到完整 benchmark 统计。
- 中止前进度接近逐条串行处理，符合 KV capacity 不足导致 scheduler 排队/分批执行的预期。

结论：

- KV capacity 对单并发 40k 影响不明显。
- KV capacity 对多并发 40k 非常关键；70950 tokens 基本只能容纳 1 条 40k 请求稳定运行。
- 原始纯 4TP 的 187k capacity 理论上可容纳约 4 条 40k 请求，TP+CP 的 71k capacity 更容易在长上下文并发下受限。

## 8. Nsight Systems Profiling 建议

如果要抓 GPU kernel timeline，`nsys` 应该包住服务端进程，而不是只包住 benchmark 客户端。

示例：

```bash
mkdir -p /home/lizihan/cuda_study/qwen3_32b_sglang_4tp/nsys

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
CC=/home/lizihan/bin/gcc-python312 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/usr/local/cuda-12.8/bin/nsys profile \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=graph \
  --sample=none \
  --cpuctxsw=none \
  -o /home/lizihan/cuda_study/qwen3_32b_sglang_4tp/nsys/qwen3_32b_4tp_sglang \
  /home/lizihan/sglang_cp_env/bin/python -m sglang.launch_server \
    --model-path /DaTa/lizihan/weight/Qwen3-32B \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.88 \
    --context-length 32768 \
    --disable-radix-cache \
    --attention-backend triton \
    --sampling-backend pytorch \
    --disable-custom-all-reduce \
    --skip-server-warmup \
    --log-level info
```

注意：

- 该方式会把权重加载、CUDA graph capture 也录入，trace 文件可能很大。
- 更干净的方式是用 `--capture-range=cudaProfilerApi` 或 NVTX range 控制采集窗口，但需要服务端在 benchmark 前后调用 profiler start/stop API 或在代码里打 NVTX range。

Profiling 时重点关注：

- attention kernel 时间；
- GEMM kernel 时间；
- NCCL all-reduce / all-gather 时间；
- CPU 调度间隙；
- CUDA graph replay 是否生效；
- chunked prefill 的切分粒度。

## 9. 常见问题与解决方法

### 9.1 localhost 请求被代理劫持

现象：

```text
Squid ERROR 403
http://127.0.0.1:30000/model_info
```

原因：环境中设置了 `http_proxy/https_proxy`，SGLang 服务端内部访问 `127.0.0.1` 也走代理。

解决：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost ...
```

启动服务时同时加：

```bash
--skip-server-warmup
```

### 9.2 FlashInfer 与 RTX 5090 / CUDA 12.8 兼容问题

现象：

```text
SM 12.x requires CUDA >= 12.9
FlashInfer requires GPUs with sm75 or higher
```

解决：

```bash
--attention-backend triton
--sampling-backend pytorch
```

### 9.3 sglang-kernel CUDA 版本不匹配

现象：

```text
ImportError: libnvrtc.so.13: cannot open shared object file
```

解决：

```bash
/home/lizihan/sglang_cp_env/bin/python -m pip install \
  --force-reinstall --no-deps \
  --index-url https://docs.sglang.ai/whl/cu129/ \
  'sglang-kernel==0.4.3+cu129'
```

### 9.4 缺少 FlashInfer 包

现象：

```text
ModuleNotFoundError: No module named 'flashinfer'
```

解决：只安装包本身，不安装依赖，避免 pip 顺手把 torch 升到 CUDA 13。

```bash
/home/lizihan/sglang_cp_env/bin/python -m pip install --no-deps flashinfer-python
```

### 9.5 Triton 编译缺 Python headers

现象：

```text
fatal error: Python.h: No such file or directory
fatal error: x86_64-linux-gnu/python3.12/pyconfig.h: No such file or directory
```

原因：系统没有安装 `python3.12-dev`，且当前用户没有免密 sudo。

解决：把 `python3.12-dev` 和 `libpython3.12-dev` 的 deb 解包到用户目录，然后用 gcc wrapper 指向本地 headers。

当前 wrapper：

```bash
/home/lizihan/bin/gcc-python312
```

内容：

```bash
#!/bin/sh
exec /usr/bin/gcc \
  -I/home/lizihan/local_python_headers/usr/include \
  -I/home/lizihan/local_python_headers/usr/include/python3.12 \
  -I/home/lizihan/local_python_headers/usr/include/x86_64-linux-gnu/python3.12 \
  "$@"
```

启动服务时设置：

```bash
CC=/home/lizihan/bin/gcc-python312
```

### 9.6 `random` benchmark 会联网下载 ShareGPT

现象：

```text
Network is unreachable
HEAD https://huggingface.co/datasets/anon8231489123/ShareGPT...
```

解决：使用纯本地 token id：

```bash
--dataset-name random-ids
```

### 9.7 `random-ids` 默认不是固定输入长度

现象：设置 `--random-input-len 4096` 后，总输入 token 不是 `num_prompts * 4096`。

原因：默认 `--random-range-ratio 0`，实际长度在 `1..4096` 中随机。

解决：

```bash
--random-range-ratio 1
```

### 9.8 40k `random-ids` 必须使用 `--tokenize-prompt`

现象：设置 `--random-input-len 40000` 后服务端报：

```text
The input is longer than the model's context length (40960 tokens)
```

原因：未使用 `--tokenize-prompt` 时，benchmark 构造的字符串会被 tokenizer 重新编码，实际 token 数超过 40000。

解决：

```bash
--tokenize-prompt
```

### 9.9 停服时出现 KeyboardInterrupt 栈

现象：`Ctrl-C` 或 `kill` 停止 SGLang 时，scheduler/detokenizer 子进程打印 `KeyboardInterrupt` 栈。

结论：这是多进程服务停止过程中的常见日志，只要进程退出且 GPU 显存释放即可。

## 10. 后续建议

- 当前环境下，40k 单请求 prefill 推荐使用纯 4TP。
- 4096 token prefill 若追求吞吐与延迟平衡，推荐并发 4；若追求最低延迟，推荐并发 1。
- 若继续研究 TP+CP，应优先使用 nsys 对比 NCCL 通信和 attention kernel 时间，验证 CP 通信是否为主要瓶颈。
- 对长上下文多并发，应明确区分计算瓶颈和 KV capacity 瓶颈；70k KV capacity 对 40k 多并发明显不足。
