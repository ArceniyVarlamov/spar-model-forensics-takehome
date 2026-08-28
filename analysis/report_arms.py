"""New 2x2 arms against the mentor's own conditions, on the same models.

  python3 analysis/report_arms.py
"""
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import mannwhitneyu
from analysis.core import run_dirs, load, raw, COND
from analysis.answers import parse_answer

NEW = Path(__file__).resolve().parents[1] / "runs_new"
ARMS = ("above_good_repl", "bet_neutral", "hidden_above", "hidden_below")


def arm_answers(model, arm):
    p = NEW / model / f"{arm}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    a = [parse_answer(r.get("content") or "") for r in d["rows"] if "error" not in r]
    return np.array([x for x in a if x is not None], dtype=float), d["threshold"], len(d["rows"])


def main():
    dirs = run_dirs()
    for model in sorted(p.name for p in NEW.glob("*") if p.is_dir()):
        key = next((k for k in dirs if k == model), None)
        print("\n" + "=" * 92)
        print(f"{model}")
        print("=" * 92)
        rows = []
        if key:
            R = load(dirs[key])
            thr = R["thr"]
            for c in COND:
                a = [parse_answer(r.get("content") or "") for r in raw(dirs[key], c) if "error" not in r]
                a = np.array([x for x in a if x is not None], dtype=float)
                rows.append((f"{c} (ментор)", a, thr))
        for arm in ARMS:
            got = arm_answers(model, arm)
            if got is None:
                print(f"  {arm}: нет данных")
                continue
            a, thr, n_all = got
            rows.append((f"{arm} (новое)", a, thr))
        print(f"{'условие':26s} {'n':>4s} {'== порог':>9s} {'медиана/порог':>14s} {'>порог':>7s} {'мода':>16s}")
        for label, a, thr in rows:
            if not len(a):
                continue
            mode, cnt = Counter(a).most_common(1)[0]
            print(f"{label:26s} {len(a):4d} {(a == thr).mean():9.2f} {np.median(a)/thr:14.2f} "
                  f"{(a > thr).mean():7.2f} {mode/1e6:>11.2f}M×{cnt}")
        d = dict((l, a) for l, a, _ in rows)
        if "hidden_above (новое)" in d and "hidden_below (новое)" in d:
            u = mannwhitneyu(d["hidden_above (новое)"], d["hidden_below (новое)"], alternative="two-sided")
            hi, lo = np.median(d["hidden_above (новое)"]), np.median(d["hidden_below (новое)"])
            thr = rows[0][2]
            print(f"\n  B↑ vs B↓ (якоря нет, мотив есть): медианы {hi/thr:.2f} и {lo/thr:.2f} порога, "
                  f"зазор {(hi-lo)/thr:+.3f}, MW p={u.pvalue:.1e}")
        if "bet_neutral (новое)" in d:
            base = d.get("baseline (ментор)")
            thr = rows[0][2]
            print(f"  A (число названо, ценности нет): ответов ровно на пороге "
                  f"{(d['bet_neutral (новое)'] == thr).mean():.2f}"
                  + (f", в baseline {(base == thr).mean():.2f}" if base is not None else ""))


if __name__ == "__main__":
    main()
