import json
import random
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor


MODEL = "/DaTa/lizihan/models/Qwen2.5-7B-Instruct"
PREFILLS = [
    ("http://127.0.0.1:32100", 9110),
    ("http://127.0.0.1:32101", 9111),
    ("http://127.0.0.1:32102", 9112),
]
DECODE = "http://127.0.0.1:32110"


def post_json(url, payload, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "latency_s": time.perf_counter() - t0,
                "body": body[:500],
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "latency_s": time.perf_counter() - t0,
            "body": body[:1000],
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "latency_s": time.perf_counter() - t0,
            "body": repr(exc),
        }


def build_messages(conv_id, turn):
    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. Answer with one short token when possible.",
        }
    ]
    for t in range(turn):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Conversation {conv_id}. Shared prefix for cache test. "
                    f"Facts: alpha={conv_id}, beta={t}. "
                    "Repeat the stable context silently and answer OK."
                ),
            }
        )
        if t < turn - 1:
            messages.append({"role": "assistant", "content": "OK"})
    return messages


def choose_prefill(policy, conv_id, seq_no):
    if policy == "random":
        return random.randint(0, len(PREFILLS) - 1)
    if policy == "round_robin":
        return seq_no % len(PREFILLS)
    if policy == "sticky":
        return conv_id % len(PREFILLS)
    raise ValueError(policy)


def run_one(policy, conv_id, turn, seq_no):
    prefill_idx = choose_prefill(policy, conv_id, seq_no)
    prefill_url, bootstrap_port = PREFILLS[prefill_idx]
    bootstrap_room = random.randint(0, 2**63 - 1)
    host = urllib.parse.urlparse(prefill_url).hostname
    payload = {
        "model": MODEL,
        "messages": build_messages(conv_id, turn),
        "temperature": 0,
        "max_tokens": 1,
        "stream": False,
        "bootstrap_host": host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": bootstrap_room,
    }

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        prefill_fut = pool.submit(
            post_json, f"{prefill_url}/v1/chat/completions", payload
        )
        decode_fut = pool.submit(post_json, f"{DECODE}/v1/chat/completions", payload)
        prefill_ret = prefill_fut.result()
        decode_ret = decode_fut.result()

    return {
        "policy": policy,
        "conv_id": conv_id,
        "turn": turn,
        "prefill_idx": prefill_idx,
        "ok": prefill_ret["ok"] and decode_ret["ok"],
        "prefill_status": prefill_ret["status"],
        "decode_status": decode_ret["status"],
        "latency_s": time.perf_counter() - t0,
        "prefill_latency_s": prefill_ret["latency_s"],
        "decode_latency_s": decode_ret["latency_s"],
        "decode_body": decode_ret["body"],
        "prefill_body": prefill_ret["body"],
    }


def run_policy(policy, conv_ids):
    results = []
    seq_no = 0
    for turn in range(1, 5):
        for conv_id in conv_ids:
            result = run_one(policy, conv_id, turn, seq_no)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
            seq_no += 1
    latencies = [item["latency_s"] for item in results]
    summary = {
        "policy": policy,
        "requests": len(results),
        "ok": sum(1 for item in results if item["ok"]),
        "failed": sum(1 for item in results if not item["ok"]),
        "avg_latency_s": statistics.mean(latencies),
        "p50_latency_s": statistics.median(latencies),
        "max_latency_s": max(latencies),
        "min_latency_s": min(latencies),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    return summary


def main():
    random.seed(20260707)
    summaries = [
        run_policy("random", range(200, 205)),
        run_policy("sticky", range(300, 305)),
    ]
    print(json.dumps({"summaries": summaries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
