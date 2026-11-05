"""Latency benchmark for HSTC + v2a — Fix #3 for NEXUS v5.2 attack.

Measures per-call p50/p95/p99 at 1k/4k/16k input sizes.
Hardware: as reported at runtime.
"""
from __future__ import annotations

import json
import pathlib
import platform
import statistics as st
import time

from runner_r3 import extractive_compress as v2a
from compressor_hstc import hstc_compress

OUT = pathlib.Path(__file__).parent.parent / "results" / "latency_bench.json"


def gen_article(target_chars: int) -> str:
    seed = "The President announced yesterday a new economic plan. Officials say it takes effect in 2026. Critics argue it is too expensive and will worsen the deficit. Supporters point to projected job growth of 4 million by 2027. The plan comes after months of debate in Congress. "
    out = ""
    while len(out) < target_chars:
        out += seed
    return out[:target_chars]


def bench(fn, text: str, runs: int = 200) -> dict:
    """Return per-call latency distribution in ms."""
    # warmup
    for _ in range(5):
        fn(text)
    times = []
    for _ in range(runs):
        t = time.perf_counter_ns()
        fn(text)
        times.append((time.perf_counter_ns() - t) / 1e6)
    times.sort()
    n = len(times)
    return {
        "runs": n,
        "p50_ms": times[n // 2],
        "p95_ms": times[int(n * 0.95)],
        "p99_ms": times[int(n * 0.99)],
        "mean_ms": st.mean(times),
    }


def main():
    print(f"Hardware: {platform.machine()} / {platform.system()} / Python {platform.python_version()}\n")
    results = {
        "hardware": f"{platform.machine()} / {platform.system()} / Python {platform.python_version()}",
        "runs_per_cell": 200,
        "bench": {},
    }

    sizes = [1000, 4000, 16000]
    print(f"{'input (chars)':14s} {'method':8s} {'p50_ms':>8s} {'p95_ms':>8s} {'p99_ms':>8s} {'mean_ms':>8s}")
    print("-" * 60)
    for size in sizes:
        text = gen_article(size)
        for name, fn in [("v2a", v2a), ("hstc", hstc_compress)]:
            r = bench(fn, text)
            print(f"{size:<14d} {name:8s} {r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} {r['p99_ms']:>8.2f} {r['mean_ms']:>8.2f}")
            results["bench"][f"{name}_{size}ch"] = r

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n✓ Written to {OUT}")


if __name__ == "__main__":
    main()
