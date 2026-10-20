"""
Offline Evaluation Suite.
Measures real routing accuracy, guardrail recall, and cache hit rates across tests/eval_set.json.

Usage:
    python scripts/run_eval.py
"""
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.orchestrator import handle_message

EVAL_SET_PATH = ROOT_DIR / "tests" / "eval_set.json"
REPORT_PATH = ROOT_DIR / "data" / "eval_report.json"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_evaluation():
    if not EVAL_SET_PATH.exists():
        print(f"Error: Eval dataset not found at {EVAL_SET_PATH}")
        sys.exit(1)

    eval_items = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    print(f"\n========================================================")
    print(f"🔬 Running Support System Evaluation ({len(eval_items)} benchmark cases)")
    print(f"========================================================\n")

    routing_correct = 0
    routing_total = 0

    injection_blocked = 0
    injection_total = 0

    clean_queries_passed = 0
    clean_total = 0

    latencies_ms = []
    cache_hits = 0
    results = []

    for item in eval_items:
        query = item["query"]
        expected_intent = item.get("expected_intent")
        is_injection = item.get("is_injection", False)

        t0 = time.perf_counter()
        resp = handle_message(query)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency_ms)

        if resp.cache_hit:
            cache_hits += 1

        is_blocked = resp.blocked_at is not None

        # Guardrail metrics
        if is_injection:
            injection_total += 1
            if is_blocked:
                injection_blocked += 1
                status = "PASS (Blocked)"
            else:
                status = "FAIL (Leaked Injection)"
        else:
            clean_total += 1
            if not is_blocked:
                clean_queries_passed += 1
                # Routing metrics
                routing_total += 1
                if resp.intent == expected_intent:
                    routing_correct += 1
                    status = "PASS"
                else:
                    status = f"FAIL (Got {resp.intent}, expected {expected_intent})"
            else:
                status = "FAIL (Falsely Blocked Safe Query)"

        print(f"[{status:^24}] \"{query[:45]:<45}\" -> intent={resp.intent or 'BLOCKED'} ({latency_ms:.0f}ms)")
        results.append({
            "query": query,
            "status": status,
            "expected_intent": expected_intent,
            "actual_intent": resp.intent,
            "blocked_at": resp.blocked_at,
            "cache_hit": resp.cache_hit,
            "latency_ms": round(latency_ms, 2),
        })

    routing_acc = (routing_correct / routing_total * 100.0) if routing_total else 0.0
    guardrail_recall = (injection_blocked / injection_total * 100.0) if injection_total else 0.0
    guardrail_precision = (injection_blocked / (injection_blocked + (clean_total - clean_queries_passed)) * 100.0) if (injection_blocked + (clean_total - clean_queries_passed)) else 100.0

    latencies_sorted = sorted(latencies_ms)
    p50_ms = latencies_sorted[len(latencies_sorted) // 2]
    p95_ms = latencies_sorted[int(len(latencies_sorted) * 0.95)]

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": len(eval_items),
        "routing_accuracy_pct": round(routing_acc, 2),
        "guardrail_recall_pct": round(guardrail_recall, 2),
        "guardrail_precision_pct": round(guardrail_precision, 2),
        "cache_hit_rate_pct": round(cache_hits / len(eval_items) * 100.0, 2),
        "latency_p50_ms": round(p50_ms, 2),
        "latency_p95_ms": round(p95_ms, 2),
        "cases": results,
    }

    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--------------------------------------------------------")
    print(f"📊 Evaluation Summary:")
    print(f"   Routing Accuracy:    {routing_acc:.1f}% ({routing_correct}/{routing_total})")
    print(f"   Guardrail Recall:    {guardrail_recall:.1f}% ({injection_blocked}/{injection_total})")
    print(f"   Guardrail Precision: {guardrail_precision:.1f}%")
    print(f"   Latency p50 / p95:   {p50_ms:.0f}ms / {p95_ms:.0f}ms")
    print(f"   Detailed report:     {REPORT_PATH}")
    print("--------------------------------------------------------\n")


if __name__ == "__main__":
    run_evaluation()
