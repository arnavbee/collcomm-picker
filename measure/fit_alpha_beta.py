"""Fit T = alpha + bytes/BW to a measured all-reduce sweep, separately for the small and large
message regimes, and print the numbers picker.py needs.

    python3 fit_alpha_beta.py result.json
"""
import json, sys


def linfit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope                    # intercept (us), slope (us per byte)


d = json.load(open(sys.argv[1]))
rows = d["rows"]
print(f"{d['device']} | {d['backend']} | world={d['world']} | torch {d['torch']}")
small = [r for r in rows if r["bytes"] <= 64 * 1024]
large = [r for r in rows if r["bytes"] >= 4 * 1024 * 1024]
a_small = sum(r["median_us"] for r in rows if r["bytes"] <= 1024) / max(1, len([r for r in rows if r["bytes"] <= 1024]))
_, slope = linfit([r["bytes"] for r in large], [r["median_us"] for r in large])
print(f"small-message latency (<=1 KB, median of medians): {a_small:8.1f} us   <- alpha-ish, includes {d['world']}-rank sync")
print(f"large-message slope: {slope:.6f} us/B  ->  effective bandwidth {1e6 / slope / 1e9:6.2f} GB/s (algorithm bandwidth, all-reduce)")
knee = next((r["bytes"] for r in rows if r["median_us"] > 2 * a_small), None)
print(f"latency doubles vs small-message floor at ~{knee} B: below this, adding bandwidth does not help")
