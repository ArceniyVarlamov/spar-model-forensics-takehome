"""Markdown tables for the write-up, generated from the data.

Numbers in paper/DRAFT.md are pasted from here, never typed by hand — an earlier
draft drifted from the recomputed values twice.

  python3 analysis/tables_md.py shift     # median answer by condition
  python3 analysis/tables_md.py first     # first in-trace estimate vs answer
  python3 analysis/tables_md.py spread    # IQR compression
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import mannwhitneyu
from analysis.core import run_dirs, load, raw, COND
from analysis.answers import parse_answer

NAME = {"claude-opus-4-7": "claude-opus-4-7", "deepseek-v4-flash-0731": "deepseek-v4-flash",
        "deepseek-v4-pro-0813": "deepseek-v4-pro", "glm-5p2": "glm-5.2",
        "qwen3p8-2p4t-a95b": "qwen3.8", "qwen3.5-122b-a10b": "qwen3.5-122b"}


def answers(d, c):
    v = [parse_answer(r.get("content") or "") for r in raw(d, c) if "error" not in r]
    return np.array([x for x in v if x is not None], dtype=float)


def rows():
    for name, d in run_dirs().items():
        R = load(d)
        yield NAME.get(name, name), d, R, R["thr"]


def shift():
    out = []
    for label, d, R, thr in rows():
        a = {c: answers(d, c) / thr for c in COND}
        p = mannwhitneyu(a["above_good"], a["below_good"], alternative="two-sided").pvalue
        m = [np.median(a[c]) for c in COND]
        out.append((m[2] - m[1], f"| {label} | {m[0]:.2f} → {m[1]:.2f} → {m[2]:.2f} | "
                                 f"{m[2]-m[1]:+.3f} | {p:.1e} |"))
    print("| model | median answer / threshold: baseline → below → above | shift | p |")
    print("|---|---|---|---|")
    for _, line in sorted(out, reverse=True):
        print(line)


def first():
    print("| model | first estimate: below → above | p | final answer: below → above | p |")
    print("|---|---|---|---|---|")
    out = []
    for label, d, R, thr in rows():
        f = {c: np.array([t[0] for t in R["traj"][c] if isinstance(t, list) and t], dtype=float) / thr
             for c in ("below_good", "above_good")}
        a = {c: answers(d, c) / thr for c in ("below_good", "above_good")}
        pf = mannwhitneyu(f["above_good"], f["below_good"], alternative="two-sided").pvalue
        pa = mannwhitneyu(a["above_good"], a["below_good"], alternative="two-sided").pvalue
        if pf < 0.05:
            out.append((pf, f"| {label} | {np.median(f['below_good']):.2f} → "
                            f"{np.median(f['above_good']):.2f} | {pf:.1e} | "
                            f"{np.median(a['below_good']):.2f} → {np.median(a['above_good']):.2f} | {pa:.1e} |"))
    for _, line in sorted(out):
        print(line)


def spread():
    print("| model | IQR width / threshold: baseline → below → above | below ÷ baseline |")
    print("|---|---|---|")
    rb = []
    for label, d, R, thr in rows():
        w = []
        for c in COND:
            q1, q3 = np.percentile(answers(d, c) / thr, [25, 75])
            w.append(q3 - q1)
        rb.append(w[1] / w[0])
        print(f"| {label} | {w[0]:.2f} → {w[1]:.2f} → {w[2]:.2f} | {w[1]/w[0]:.2f}× |")
    print(f"\nmedian compression in below-favoured: {np.median(rb):.2f}× of baseline")


if __name__ == "__main__":
    {"shift": shift, "first": first, "spread": spread}[sys.argv[1]]()
