"""Statistics behind section 4: dispersion, contrasts and power for the 2x2 arms.

report_arms.py prints the cells; this prints the numbers the argument rests on —
IQR ratios against baseline with bootstrap intervals, the pairwise contrasts, and
what n each contrast would need.

  python3 analysis/arms_stats.py [B] [seed]        # defaults 4000, 0
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import mannwhitneyu
from analysis.core import run_dirs, load, raw, COND
from analysis.answers import parse_answer

NEW = Path(__file__).resolve().parents[1] / "runs_new"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
LABEL = {"above_good_repl": "A_repl (backend control)", "bet_neutral": "A (number, no value)",
         "hidden_above": "B^ (value, no number)", "hidden_below": "Bv (value, no number)"}


def iqr(a):
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def cell_new(model, arm, thr):
    p = NEW / model / f"{arm}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ok = [r for r in d["rows"] if "error" not in r]
    a = np.array([x for x in (parse_answer(r.get("content") or "") for r in ok) if x is not None], float) / thr
    ln = np.array([len(r.get("reasoning") or "") for r in ok], float)
    return (a, ln) if a.size else None


def cell_his(d, cond, thr):
    rows = [r for r in raw(d, cond) if "error" not in r]
    a = np.array([x for x in (parse_answer(r.get("content") or "") for r in rows) if x is not None], float) / thr
    ln = np.array([len(r.get("reasoning") or "") for r in rows], float)
    return a, ln


def ratio_ci(rng, arm, base):
    b = np.array([iqr(rng.choice(arm, arm.size, True)) / max(iqr(rng.choice(base, base.size, True)), 1e-12)
                  for _ in range(B)])
    return iqr(arm) / iqr(base), *np.percentile(b, [2.5, 97.5])


def power(rng, x, y, n, reps=1000):
    """Share of resampled experiments at n per arm that reject at 0.05."""
    hit = 0
    for _ in range(reps):
        p = mannwhitneyu(rng.choice(x, n, True), rng.choice(y, n, True), alternative="two-sided").pvalue
        hit += p < 0.05
    return hit / reps


def main():
    rng = np.random.default_rng(SEED)
    dirs = run_dirs()
    for model in sorted(p.name for p in NEW.glob("*") if p.is_dir()):
        key = next((k for k in dirs if k == model), None)
        print("\n" + "=" * 96 + f"\n{model}\n" + "=" * 96)
        if not key:
            print("no match in your grid — new arms only")
        cells = {}
        if key:
            R = load(dirs[key])
            thr = R["thr"]
            for c in COND:
                cells[c] = cell_his(dirs[key], c, thr)
        else:
            thr = json.loads((NEW / model / "bet_neutral.json").read_text())["threshold"]
        for arm in LABEL:
            got = cell_new(model, arm, thr)
            if got:
                cells[arm] = got

        base = cells.get("baseline", (None, None))[0]
        print(f"{'condition':28s} {'n':>4s} {'median/thr':>14s} {'IQR/thr':>10s} "
              f"{'IQR / base':>22s} {'median length':>14s}")
        for name, (a, ln) in cells.items():
            r = ""
            if base is not None and name != "baseline" and a.size >= 5:
                pt, lo, hi = ratio_ci(rng, a, base)
                r = f"{pt:.2f} [{lo:.2f}, {hi:.2f}]"
            print(f"{LABEL.get(name, name + ' (yours)'):28s} {a.size:4d} {np.median(a):14.2f} "
                  f"{iqr(a):10.2f} {r:>22s} {np.median(ln):14,.0f}")

        def contrast(x, y, lab):
            if x not in cells or y not in cells:
                return
            a, b = cells[x][0], cells[y][0]
            if a.size < 3 or b.size < 3:
                return
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            print(f"  {lab:44s} MW p={p:.2f}")

        print()
        contrast("above_good_repl", "above_good", "backend control vs your above")
        contrast("hidden_above", "hidden_below", "B^ vs Bv (no number)")
        contrast("bet_neutral", "baseline", "A vs baseline")
        contrast("bet_neutral", "above_good", "A vs your above")

        if "hidden_above" in cells and "hidden_below" in cells:
            for n in (40, 400):
                print(f"  power, B^ vs Bv at n={n:<4d}                             {power(rng, cells['hidden_above'][0], cells['hidden_below'][0], n):.2f}")
        if "below_good" in cells and "above_good" in cells:
            print(f"  power, your below vs above at n=40                      {power(rng, cells['below_good'][0], cells['above_good'][0], 40):.2f}")
        if "bet_neutral" in cells and base is not None:
            for n in (40, 200, 400, 800):
                print(f"  power, A vs baseline at n={n:<4d}                        {power(rng, cells['bet_neutral'][0], base, n):.2f}")


if __name__ == "__main__":
    main()
