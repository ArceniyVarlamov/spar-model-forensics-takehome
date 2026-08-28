"""How much of the panel's model ordering the data behind MRF actually supports.

MRF is a difference of two medians over ~100 rollouts each. This resamples the
per-rollout drifts to put an interval on it, then asks the two questions the
write-up makes claims about: how many model pairs are actually separated, and
whether MRF tracks the answer-level shift the bet pays out on.

  python3 analysis/bootstrap_mrf.py [B] [seed]      # defaults 10000, 0
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import spearmanr
from analysis.core import run_dirs, load, raw, valid, drift_per_rollout
from analysis.answers import parse_answer

B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def drifts(traj, thr, cond):
    kept = valid(traj.get(cond, []), thr)
    return drift_per_rollout(kept, thr) if kept else np.array([])


def answers(d, c, thr):
    a = [parse_answer(r.get("content") or "") for r in raw(d, c) if "error" not in r]
    return np.array([x for x in a if x is not None], dtype=float) / thr


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for name, d in run_dirs().items():
        R = load(d)
        thr = R["thr"]
        a, b = drifts(R["traj"], thr, "above_good"), drifts(R["traj"], thr, "below_good")
        point = float(np.median(a) - np.median(b))
        boot = np.array([
            np.median(rng.choice(a, a.size, replace=True))
            - np.median(rng.choice(b, b.size, replace=True))
            for _ in range(B)
        ])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        shift = float(np.median(answers(d, "above_good", thr)) - np.median(answers(d, "below_good", thr)))
        rows.append({"model": name, "mrf": point, "lo": float(lo), "hi": float(hi),
                     "p_pos": float((boot > 0).mean()), "n_above": a.size, "n_below": b.size,
                     "shift": shift, "reported": R["factor"]["motivated_reasoning_factor"]})

    print(f"B={B} replicates, seed={SEED}, resampling per-rollout drifts within each condition\n")
    print(f"{'model':24s} {'MRF':>8s} {'95% CI':>20s} {'P(MRF>0)':>9s} {'n above/below':>14s} {'shift':>8s}")
    for r in rows:
        print(f"{r['model']:24s} {r['mrf']:+8.4f} "
              f"{'[' + format(r['lo'], '+.3f') + ', ' + format(r['hi'], '+.3f') + ']':>20s} "
              f"{r['p_pos']:9.2f} {str(r['n_above']) + '/' + str(r['n_below']):>14s} {r['shift']:+8.3f}")

    pairs = list(combinations(rows, 2))
    sep = [(x['model'], y['model']) for x, y in pairs if x['hi'] < y['lo'] or y['hi'] < x['lo']]
    straddle = [r['model'] for r in rows if r['lo'] < 0 < r['hi']]
    strong = [r['model'] for r in rows if r['p_pos'] >= 0.9]

    print(f"\nmodel pairs with non-overlapping 95% intervals: {len(sep)} of {len(pairs)}")
    for s in sep:
        print(f"  {s[0]} vs {s[1]}")
    print(f"intervals covering zero: {len(straddle)} — {', '.join(straddle)}")
    print(f"models with P(MRF>0) >= 0.9: {len(strong)} — {', '.join(strong)}")

    rho, p = spearmanr([r['mrf'] for r in rows], [r['shift'] for r in rows])
    print(f"\nrank correlation of MRF with the answer-level shift: rho={rho:+.2f}, p={p:.2f}, n={len(rows)}")

    # the same treatment for the answer-level shift — how much more precise the
    # same rollouts are when measured where the bet actually pays out
    print("\nfor comparison — shift of the median answer, bootstrapped the same way")
    print(f"{'model':24s} {'shift':>8s} {'95% CI':>20s}")
    ar = []
    for name, d in run_dirs().items():
        R = load(d)
        thr = R["thr"]
        av, bv = answers(d, "above_good", thr), answers(d, "below_good", thr)
        boot = np.array([np.median(rng.choice(av, av.size, True)) - np.median(rng.choice(bv, bv.size, True))
                         for _ in range(B)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ar.append({"model": name, "lo": float(lo), "hi": float(hi)})
        print(f"{name:24s} {np.median(av) - np.median(bv):+8.3f} "
              f"{'[' + format(lo, '+.3f') + ', ' + format(hi, '+.3f') + ']':>20s}")
    a_sep = sum(1 for x, y in combinations(ar, 2) if x['hi'] < y['lo'] or y['hi'] < x['lo'])
    a_nz = sum(1 for r in ar if not (r['lo'] < 0 < r['hi']))
    m_nz = sum(1 for r in rows if not (r['lo'] < 0 < r['hi']))
    print(f"\npairs separated: MRF {len(sep)}/{len(pairs)}, answer shift {a_sep}/{len(pairs)}")
    print(f"intervals clear of zero: MRF {m_nz}/10, answer shift {a_nz}/10")


if __name__ == "__main__":
    main()
