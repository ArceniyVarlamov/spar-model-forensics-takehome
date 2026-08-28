"""What the bet actually pays out on: the visible answer, not the drift.

MRF measures movement inside the trace. The donation is decided by the final
number. This module computes both on the same rollouts so they can be compared.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import mannwhitneyu
from analysis.core import run_dirs, load, raw, COND
from analysis.answers import parse_answer


def per_model(name, d):
    R = load(d)
    thr = R["thr"]
    out = {"model": name, "threshold": thr, "cond": {}}
    for c in COND:
        rows = raw(d, c)
        traj = R["traj"][c]
        recs = []
        for i, r in enumerate(rows):
            if "error" in r:
                continue
            ans = parse_answer(r.get("content") or "")
            t = traj[i] if i < len(traj) and isinstance(traj[i], list) else None
            recs.append({
                "ans": ans,
                "first": t[0] if t else None,
                "last": t[-1] if t else None,
                "len": len(t) if t else 0,
                "chars": len(r.get("reasoning") or ""),
            })
        a = np.array([r["ans"] for r in recs if r["ans"] is not None], dtype=float)
        f = np.array([r["first"] for r in recs if r["first"] is not None], dtype=float)
        l = np.array([r["last"] for r in recs if r["last"] is not None], dtype=float)
        pair = [(r["last"], r["ans"]) for r in recs
                if r["last"] is not None and r["ans"] is not None]
        out["cond"][c] = {
            "n": len(recs),
            "n_ans": len(a),
            "ans_med": float(np.median(a)) if len(a) else None,
            "p_above": float((a > thr).mean()) if len(a) else None,
            "first_med": float(np.median(f)) if len(f) else None,
            "last_med": float(np.median(l)) if len(l) else None,
            "handoff_med": float(np.median([(y - x) / thr for x, y in pair])) if pair else None,
            "handoff_moved": float(np.mean([abs(y - x) / max(x, 1) > 0.01 for x, y in pair])) if pair else None,
            "answers": a,
            "firsts": f,
        }
    return out


def main():
    rows = [per_model(n, d) for n, d in run_dirs().items()]
    hdr = (f"{'model':22s} {'thr':>11s} | "
           f"{'P(ans>thr) base/below/above':>30s} | {'median ans / thr':>26s} | {'p(MW)':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        c = r["cond"]
        p = [c[x]["p_above"] for x in COND]
        m = [c[x]["ans_med"] / r["threshold"] for x in COND]
        u = mannwhitneyu(c["above_good"]["answers"], c["below_good"]["answers"],
                         alternative="two-sided")
        print(f"{r['model']:22s} {r['threshold']:>11,.0f} | "
              f"{p[0]:>9.2f} {p[1]:>9.2f} {p[2]:>9.2f} | "
              f"{m[0]:>8.2f} {m[1]:>8.2f} {m[2]:>8.2f} | {u.pvalue:8.1e}")
    print()
    print(f"{'model':22s} | {'median FIRST in-trace / thr':>28s} | {'median LAST in-trace / thr':>27s} | {'answer-minus-last (thr units)':>30s}")
    for r in rows:
        c = r["cond"]
        fm = [c[x]["first_med"] / r["threshold"] for x in COND]
        lm = [c[x]["last_med"] / r["threshold"] for x in COND]
        ho = [c[x]["handoff_med"] for x in COND]
        print(f"{r['model']:22s} | {fm[0]:>8.2f} {fm[1]:>8.2f} {fm[2]:>8.2f} | "
              f"{lm[0]:>8.2f} {lm[1]:>8.2f} {lm[2]:>8.2f} | {ho[0]:>+9.3f} {ho[1]:>+9.3f} {ho[2]:>+9.3f}")


if __name__ == "__main__":
    main()
