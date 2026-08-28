"""Parser vs the mentor's estimate judge, on the one condition where both exist."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from analysis.core import run_dirs, raw
from analysis.answers import parse_answer

tot = ok = 0
for name, d in run_dirs().items():
    est = json.loads((d / "estimates.json").read_text())["baseline"]
    for i, r in enumerate(raw(d, "baseline")):
        if "error" in r or i >= len(est) or est[i] is None:
            continue
        p = parse_answer(r.get("content") or "")
        tot += 1
        ok += p is not None and abs(p - est[i]) / max(est[i], 1) < 1e-6
print(f"answer parser vs judge (baseline): {ok}/{tot} = {ok/tot:.1%}")
