"""How much of the answer distribution sits exactly on the number in the prompt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis.core import run_dirs, load, raw, COND
from analysis.answers import parse_answer


def answers(d, c):
    a = [parse_answer(r.get("content") or "") for r in raw(d, c) if "error" not in r]
    return np.array([x for x in a if x is not None], dtype=float)


def main():
    print(f"{'model':22s} {'thr':>11s} {'round?':>7s} | "
          f"{'== thr: base':>12s} {'below':>7s} {'above':>7s} | "
          f"{'win|below':>9s} {'win|above':>9s} | {'just-above (0,5%]':>17s}")
    for name, d in run_dirs().items():
        thr = load(d)["thr"]
        base, b, a = (answers(d, c) for c in COND)
        rnd = "yes" if thr % 10_000_000 == 0 else "no"
        print(f"{name:22s} {thr:>11,.0f} {rnd:>7s} | "
              f"{(base == thr).mean():12.2f} {(b == thr).mean():7.2f} {(a == thr).mean():7.2f} | "
              f"{(b <= thr).mean():9.2f} {(a > thr).mean():9.2f} | "
              f"{((a > thr) & (a <= thr * 1.05)).mean():17.2f}")


if __name__ == "__main__":
    main()
