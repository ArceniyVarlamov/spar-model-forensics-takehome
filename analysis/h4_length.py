"""H4 — is MRF partly a trace-length effect?

The bet lengthens reasoning 1.2-2.9x, and long Fermi reasoning drifts on its own
(delta_baseline != 0 for nine of ten models). If drift grew with length, MRF
would be partly an artefact of the manipulation making traces longer.

Two measurements:
  1. rank correlation between per-rollout drift and trace length, within condition
  2. MRF recomputed inside a length band both conditions occupy, defined as the
     overlap of their 10th-90th percentile ranges

  python3 analysis/h4_length.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import spearmanr
from analysis.core import run_dirs, load, raw, drift_per_rollout

BAND_LO, BAND_HI = 10, 90


def paired(d, R, cond):
    """(drift, trace length) per rollout that reaches MRF, index-aligned with raw rows."""
    thr, traj, rows = R["thr"], R["traj"][cond], raw(d, cond)
    keep, lens = [], []
    for i, r in enumerate(rows):
        if "error" in r:
            continue
        t = traj[i] if i < len(traj) and isinstance(traj[i], list) else None
        if t is None or len(t) < 2:
            continue
        keep.append(t)
        lens.append(len(r.get("reasoning") or ""))
    if not keep:
        return np.array([]), np.array([])
    return drift_per_rollout(keep, thr), np.array(lens, dtype=float)


def main():
    print(f"{'model':24s} {'cond':11s} {'n':>4s} {'rho(drift, len)':>16s} {'p':>7s}")
    worst_rho, worst_p = 0.0, 1.0
    per_model = {}
    for name, d in run_dirs().items():
        R = load(d)
        per_model[name] = {}
        for c in ("below_good", "above_good"):
            dr, ln = paired(d, R, c)
            per_model[name][c] = (dr, ln)
            rho, p = spearmanr(dr, ln)
            worst_rho, worst_p = max(worst_rho, abs(rho)), min(worst_p, p)
            print(f"{name:24s} {c:11s} {dr.size:4d} {rho:+16.3f} {p:7.2f}")
    print(f"\nмаксимум |rho| по всем моделям и условиям: {worst_rho:.3f}; минимум p: {worst_p:.2f}")

    print(f"\nMRF внутри общей полосы длин (перекрытие p{BAND_LO}-p{BAND_HI} обоих условий)")
    print(f"{'model':24s} {'MRF полный':>11s} {'MRF в полосе':>13s} {'n в полосе':>11s} {'знак':>6s}")
    same = 0
    for name, cd in per_model.items():
        (da, la), (db, lb) = cd["above_good"], cd["below_good"]
        full = float(np.median(da) - np.median(db))
        lo = max(np.percentile(la, BAND_LO), np.percentile(lb, BAND_LO))
        hi = min(np.percentile(la, BAND_HI), np.percentile(lb, BAND_HI))
        ma, mb = (la >= lo) & (la <= hi), (lb >= lo) & (lb <= hi)
        if ma.sum() < 5 or mb.sum() < 5:
            print(f"{name:24s} {full:+11.4f} {'мало данных':>13s}")
            continue
        band = float(np.median(da[ma]) - np.median(db[mb]))
        ok = (full > 0) == (band > 0)
        same += ok
        print(f"{name:24s} {full:+11.4f} {band:+13.4f} {str(int(ma.sum())) + '/' + str(int(mb.sum())):>11s} "
              f"{'тот же' if ok else 'ИНОЙ':>6s}")
    print(f"\nзнак сохранён: {same} из {len(per_model)}")


if __name__ == "__main__":
    main()
