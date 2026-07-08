#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "/DaTa/lizihan/models/Qwen2.5-7B-Instruct"


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            elapsed = time.perf_counter() - start
            return resp.status, json.loads(body), elapsed, None
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return exc.code, parsed, elapsed, str(exc)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return None, None, elapsed, repr(exc)


def context_block(kind, group_id, repeats):
    facts = [
        f"{kind} group {group_id} fact {i}: user profile, document section, "
        f"order state, retrieval chunk, and routing affinity marker {group_id}-{i}."
        for i in range(24)
    ]
    return "\n".join(facts * repeats)


def build_sessions(scenario, num_groups, turns, context_repeats):
    system = (
        "你是统一的企业助手。所有请求的第一条 system message 完全相同。"
        "请只输出一句很短的中文回答。"
    )
    sessions = {}
    for gid in range(num_groups):
        if scenario == "support_cases":
            kind = "support-ticket"
            context = context_block(kind, gid, context_repeats)
            first = (
                f"工单上下文如下，工单编号 TICKET-{gid:03d}。\n{context}\n"
                "第 1 轮问题：概括这个工单的核心诉求。"
            )
            followups = [
                "第 2 轮问题：继续基于同一工单，判断下一步处理动作。",
                "第 3 轮问题：继续基于同一工单，给出一句最终回复。",
            ]
        elif scenario == "rag_documents":
            kind = "rag-document"
            context = context_block(kind, gid, context_repeats)
            first = (
                f"检索文档如下，文档编号 DOC-{gid:03d}。\n{context}\n"
                "第 1 轮问题：根据文档回答主题是什么。"
            )
            followups = [
                "第 2 轮问题：继续基于同一文档，列出关键证据。",
                "第 3 轮问题：继续基于同一文档，给出结论。",
            ]
        elif scenario == "code_modules":
            kind = "code-module"
            context = context_block(kind, gid, context_repeats)
            first = (
                f"代码模块如下，模块编号 MOD-{gid:03d}。\n{context}\n"
                "第 1 轮问题：说明这个模块负责什么。"
            )
            followups = [
                "第 2 轮问题：继续基于同一模块，指出潜在性能风险。",
                "第 3 轮问题：继续基于同一模块，给出优化建议。",
            ]
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        prompts = [first] + followups[: turns - 1]
        sessions[f"{scenario}-{gid:03d}"] = {
            "system": system,
            "turns": prompts,
        }
    return sessions


def build_requests(scenario, num_groups, turns, repeat, context_repeats):
    sessions = build_sessions(scenario, num_groups, turns, context_repeats)
    requests = []
    for rep in range(1, repeat + 1):
        histories = {
            sid: [{"role": "system", "content": data["system"]}]
            for sid, data in sessions.items()
        }
        for turn_idx in range(turns):
            for sid, data in sessions.items():
                histories[sid].append(
                    {"role": "user", "content": data["turns"][turn_idx]}
                )
                requests.append(
                    {
                        "scenario": scenario,
                        "conversation_id": sid,
                        "turn": turn_idx + 1,
                        "repeat": rep,
                        "messages": list(histories[sid]),
                    }
                )
                histories[sid].append(
                    {
                        "role": "assistant",
                        "content": f"ACK {sid} turn {turn_idx + 1}",
                    }
                )
    return requests


def extract_usage(obj):
    usage = (obj or {}).get("usage") or {}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    return int(prompt or 0), int(cached or 0), details


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return values[idx]


def summarize(rows):
    ok = [r for r in rows if r["ok"]]
    lat = [r["latency_s"] for r in ok]
    prompt = sum(r["prompt_tokens"] for r in ok)
    cached = sum(r["cached_tokens"] for r in ok)
    return {
        "requests": len(rows),
        "ok_requests": len(ok),
        "failed_requests": len(rows) - len(ok),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_hit_rate": (cached / prompt) if prompt else 0.0,
        "latency_avg_s": statistics.mean(lat) if lat else None,
        "latency_p50_s": percentile(lat, 50),
        "latency_p90_s": percentile(lat, 90),
        "latency_p99_s": percentile(lat, 99),
    }


def run_one(endpoint, item, strategy, timeout):
    payload = {
        "model": MODEL,
        "messages": item["messages"],
        "temperature": 0,
        "max_tokens": 1,
        "stream": False,
        "return_cached_tokens_details": True,
    }
    status, obj, latency, error = post_json(endpoint, payload, timeout)
    prompt, cached, details = extract_usage(obj)
    row = {
        "strategy": strategy,
        "scenario": item["scenario"],
        "conversation_id": item["conversation_id"],
        "turn": item["turn"],
        "repeat": item["repeat"],
        "status": status,
        "ok": status == 200,
        "latency_s": latency,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_hit_rate": (cached / prompt) if prompt else 0.0,
        "cached_tokens_details": details,
        "error": error,
        "response_id": (obj or {}).get("id") if isinstance(obj, dict) else None,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--scenario", choices=["support_cases", "rag_documents", "code_modules"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-groups", type=int, default=36)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--context-repeats", type=int, default=3)
    parser.add_argument("--seed-concurrency", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    requests = build_requests(
        args.scenario, args.num_groups, args.turns, args.repeat, args.context_repeats
    )

    rows = []
    by_phase = {}
    for item in requests:
        by_phase.setdefault((item["repeat"], item["turn"]), []).append(item)

    for phase in sorted(by_phase):
        phase_items = by_phase[phase]
        workers = args.seed_concurrency if phase[1] == 1 else args.concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run_one, endpoint, item, args.strategy, args.timeout)
                for item in phase_items
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                row["request_index"] = len(rows) + 1
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(args.sleep)

    stem = f"{args.strategy}_{args.scenario}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    summary_path = out_dir / f"{stem}.summary.json"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary.update(
        {
            "strategy": args.strategy,
            "scenario": args.scenario,
            "num_groups": args.num_groups,
            "turns": args.turns,
            "repeat": args.repeat,
            "context_repeats": args.context_repeats,
            "seed_concurrency": args.seed_concurrency,
            "concurrency": args.concurrency,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
