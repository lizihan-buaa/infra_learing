#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import random
import statistics
import time
import urllib.request
import urllib.error
from pathlib import Path


MODEL = "/DaTa/lizihan/models/Qwen2.5-7B-Instruct"


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), time.perf_counter() - start, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return exc.code, parsed, time.perf_counter() - start, str(exc)
    except Exception as exc:
        return None, None, time.perf_counter() - start, repr(exc)


def post_stream(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    first_chunk_s = None
    last_obj = None
    stream_error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_line = line[5:].strip()
                if data_line == "[DONE]":
                    break
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - start
                try:
                    obj = json.loads(data_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    stream_error = obj.get("error")
                last_obj = obj
        err = json.dumps(stream_error, ensure_ascii=False) if stream_error is not None else None
        return 200, last_obj or {}, first_chunk_s, time.perf_counter() - start, err
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return exc.code, parsed, first_chunk_s, time.perf_counter() - start, str(exc)
    except Exception as exc:
        return None, None, first_chunk_s, time.perf_counter() - start, repr(exc)


def usage(obj):
    u = (obj or {}).get("usage") or {}
    details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    prompt = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    cached = int(details.get("cached_tokens") or 0)
    return prompt, cached, details


def context(kind, sid, scale):
    unique = [
        f"{kind} session {sid:04d} stable section {i:03d}: owner marker S{sid:04d}, "
        f"routing affinity payload, repeated evidence, and unchanged multi-turn prefix body."
        for i in range(86 * scale)
    ]
    shared = [
        f"{kind} shared trailer {i:03d}: common instruction template and stable benchmark wording."
        for i in range(8 * scale)
    ]
    return "\n".join(unique + shared)


def build_phases(scenario, num_sessions, scale, seed):
    system = "你是统一企业助手。所有请求的第一条 system message 完全相同。只输出一个很短的中文短句。"
    kind = {
        "support_cases": "support-ticket",
        "rag_documents": "rag-document",
        "code_modules": "code-module",
    }[scenario]
    questions = {
        "support_cases": ["概括核心诉求。", "判断下一步处理动作。", "给出最终一句回复。"],
        "rag_documents": ["回答文档主题。", "列出关键证据。", "给出一句结论。"],
        "code_modules": ["说明模块职责。", "指出潜在性能风险。", "给出一句优化建议。"],
    }[scenario]
    rng = random.Random(seed)
    phases = []
    contexts = {sid: context(kind, sid, scale) for sid in range(num_sessions)}
    for turn, q in enumerate(questions, 1):
        items = []
        phase = "seed_turn1" if turn == 1 else f"warm_turn{turn}"
        for sid in range(num_sessions):
            user = f"会话 {sid:04d} 的长上下文如下：\n{contexts[sid]}\n问题：{q}"
            items.append({
                "phase": phase,
                "turn": turn,
                "conversation_id": f"{scenario}-{sid:04d}",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            })
        rng.shuffle(items)
        phases.append(items)
    return phases


def percentile(vals, pct):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, max(0, round(pct / 100 * (len(vals) - 1))))]


def summarize(rows):
    ok = [r for r in rows if r["ok"]]
    prompt = sum(r["prompt_tokens"] for r in ok)
    cached = sum(r["cached_tokens"] for r in ok)
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    total = [r["total_latency_s"] for r in ok]
    return {
        "requests": len(rows),
        "ok_requests": len(ok),
        "failed_requests": len(rows) - len(ok),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_hit_rate": cached / prompt if prompt else 0,
        "ttft_avg_s": statistics.mean(ttft) if ttft else None,
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p90_s": percentile(ttft, 90),
        "ttft_p99_s": percentile(ttft, 99),
        "total_avg_s": statistics.mean(total) if total else None,
        "total_p50_s": percentile(total, 50),
        "total_p90_s": percentile(total, 90),
        "total_p99_s": percentile(total, 99),
    }


def run_one(endpoint, item, strategy, timeout):
    payload = {
        "model": MODEL,
        "messages": item["messages"],
        "temperature": 0,
        "max_tokens": 8,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_cached_tokens_details": True,
    }
    started = time.time()
    status, obj, ttft_s, total_s, err = post_stream(endpoint, payload, timeout)
    prompt, cached, details = usage(obj)
    return {
        "strategy": strategy,
        "phase": item["phase"],
        "turn": item["turn"],
        "conversation_id": item["conversation_id"],
        "status": status,
        "ok": status == 200 and err is None,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "ttft_s": ttft_s,
        "total_latency_s": total_s,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_hit_rate": cached / prompt if prompt else 0,
        "cached_tokens_details": details,
        "error": err,
        "response_id": (obj or {}).get("id") if isinstance(obj, dict) else None,
    }


def flush(base_url, timeout):
    return post_json(base_url.rstrip("/") + "/flush_cache", {}, timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--scenario", choices=["support_cases", "rag_documents", "code_modules"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-sessions", type=int, default=48)
    ap.add_argument("--tokens-scale", type=int, default=1)
    ap.add_argument("--seed-concurrency", type=int, default=32)
    ap.add_argument("--warm-concurrency", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--flush-cache", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.flush_cache:
        status, obj, elapsed, err = flush(args.base_url, args.timeout)
        print(json.dumps({"event": "flush_cache", "status": status, "elapsed_s": elapsed, "error": err, "response": obj}, ensure_ascii=False), flush=True)
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    rows = []
    for phase_items in build_phases(args.scenario, args.num_sessions, args.tokens_scale, args.seed)[:args.max_turns]:
        workers = args.seed_concurrency if phase_items[0]["phase"] == "seed_turn1" else args.warm_concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one, endpoint, item, args.strategy, args.timeout) for item in phase_items]
            for fut in concurrent.futures.as_completed(futs):
                row = fut.result()
                row["request_index"] = len(rows) + 1
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1)

    stem = f"{args.strategy}_{args.scenario}_same_context"
    (out / f"{stem}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = summarize(rows)
    summary.update(vars(args))
    (out / f"{stem}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    by_phase = {p: summarize([r for r in rows if r["phase"] == p]) for p in sorted({r["phase"] for r in rows})}
    (out / f"{stem}.by_phase.json").write_text(json.dumps(by_phase, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "by_phase": by_phase}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
