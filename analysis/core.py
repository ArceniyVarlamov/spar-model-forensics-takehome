"""Shared loaders + metric re-implementations for the Donation Bet corpus.

Pure functions over the mentor's run dirs. No API calls.
"""
import json, glob, os
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "vendor" / "value-leakage" / "runs"
COND = ("baseline", "below_good", "above_good")
N_GRID = 1000
OUTLIER_FACTOR = 10


def run_dirs():
    out = {}
    for d in sorted(RUNS.glob("*_2026*")):
        out[d.name.split("_2026")[0]] = d
    return out


def load(d):
    j = lambda n: json.loads((d / n).read_text())
    return {
        "traj": j("trajectories.json"),
        "thr": j("threshold.json")["threshold"],
        "factor": j("factor.json"),
        "config": j("config.json"),
    }


def raw(d, cond):
    return json.loads((d / f"{cond}.json").read_text())["rows"]


def valid(trajs, thr=None, outlier_factor=None):
    kept = [t for t in trajs if isinstance(t, list) and len(t) >= 2]
    if outlier_factor is None or thr is None:
        return kept
    lo, hi = thr / outlier_factor, thr * outlier_factor
    return [t for t in kept if all(lo <= v <= hi for v in t)]


def resample(t, n=N_GRID):
    a = np.asarray(t, dtype=float)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a)


def drift_per_rollout(trajs, thr, window=0.2):
    w = max(1, int(round(N_GRID * window)))
    out = []
    for t in trajs:
        g = resample(t)
        out.append((g[-w:].mean() - g[:w].mean()) / thr)
    return np.array(out)


def drift(trajs, thr, window=0.2, outlier_factor=None):
    kept = valid(trajs, thr, outlier_factor)
    if not kept:
        return None
    return float(np.median(drift_per_rollout(kept, thr, window)))


def mrf(traj, thr, outlier_factor=None):
    a = drift(traj.get("above_good", []), thr, outlier_factor=outlier_factor)
    b = drift(traj.get("below_good", []), thr, outlier_factor=outlier_factor)
    if a is None or b is None:
        return None
    return a - b


_NUM_HEAD = None
import re as _re
_LEADING = _re.compile(r"^\s*[*_#\s]*([0-9][0-9,\s]*(?:\.[0-9]+)?)\s*(million|billion|thousand|m|bn|b|k)?\b", _re.I)
_MULT = {"million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9, "thousand": 1e3, "k": 1e3}


def head_number(content: str):
    """Number the visible answer opens with, or None. Conservative on purpose."""
    if not isinstance(content, str):
        return None
    m = _LEADING.match(content)
    if not m:
        return None
    raw = m.group(1).replace(",", "").replace(" ", "")
    try:
        v = float(raw)
    except ValueError:
        return None
    suf = (m.group(2) or "").lower()
    if suf:
        v *= _MULT[suf]
    return v
