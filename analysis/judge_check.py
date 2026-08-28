"""Independent check of the local answer parser on the BET conditions.

Baseline validation cannot catch bet-specific failure modes: baseline answers
never quote a threshold. This runs the mentor's verbatim estimate-judge prompt
over a stratified sample of below_good/above_good rollouts and compares.

  python3 analysis/judge_check.py [n_per_cell]
"""
import json, os, re, sys, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis.core import run_dirs, load, raw
from analysis.answers import parse_answer

JUDGE_SRC = Path("mirror/value-leakage/src/value_leakage/judge.py").read_text()
PROMPT = re.search(r'NUMBER_JUDGE_PROMPT = """\\\n(.*?)"""', JUDGE_SRC, re.S).group(1)
TAG = re.compile(r"<final_estimate>\s*(.*?)\s*</final_estimate>", re.DOTALL)
BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
KEY = os.environ["QWEN_WS_API_KEY"]
MODEL = "qwen3.8-flash"


def judge(text):
    body = {"model": MODEL, "max_tokens": 64, "enable_thinking": False,
            "messages": [{"role": "user", "content": PROMPT.format(llm_text=text)}]}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())["choices"][0]["message"]["content"] or ""
            m = TAG.search(out)
            if not m:
                return None
            v = m.group(1).strip().replace(",", "")
            return None if v.upper() == "UNKNOWN" else float(v)
        except Exception:
            if attempt == 2:
                return "ERR"
    return "ERR"


def main():
    per_cell = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    rng = np.random.default_rng(20260826)
    jobs = []
    for name, d in run_dirs().items():
        thr = load(d)["thr"]
        for c in ("below_good", "above_good"):
            rows = [r for r in raw(d, c) if "error" not in r and (r.get("content") or "").strip()]
            pick = rng.choice(len(rows), size=min(per_cell, len(rows)), replace=False)
            for i in pick:
                jobs.append((name, c, thr, rows[i].get("content")))
    print(f"{len(jobs)} проверок судьёй {MODEL}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        verdicts = list(ex.map(lambda j: judge(j[3]), jobs))
    cache = [{"model": n, "cond": c, "thr": thr, "content": content,
              "judge": (None if v in (None, "ERR") else v), "status": ("ERR" if v == "ERR" else ("UNKNOWN" if v is None else "ok"))}
             for (n, c, thr, content), v in zip(jobs, verdicts)]
    out = Path(__file__).resolve().parents[1] / "artifacts" / "judge-labels.json"
    out.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"размеченный набор сохранён: {out}")

    agree = tot = err = unk = 0
    bad = []
    for (name, c, thr, content), v in zip(jobs, verdicts):
        if v == "ERR":
            err += 1
            continue
        if v is None:
            unk += 1
            continue
        p = parse_answer(content)
        tot += 1
        if p is not None and abs(p - v) / max(v, 1) < 1e-6:
            agree += 1
        else:
            bad.append((name, c, v, p, (content or "")[:110].replace("\n", " ")))
    print(f"\nсогласие парсера с судьёй на условиях со ставкой: {agree}/{tot} = {agree/max(tot,1):.1%}"
          f"  (судья UNKNOWN: {unk}, ошибок вызова: {err})")
    for b in bad[:15]:
        print(f"  {b[0]:22s} {b[1]:11s} судья={b[2]:>14,.0f} парсер="
              f"{('None' if b[3] is None else format(b[3], ',.0f')):>14s} | {b[4]!r}")


if __name__ == "__main__":
    main()
