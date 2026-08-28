"""Same arms, run on OpenRouter free models. Patient: upstream 429 is the norm.

  python3 analysis/run_arms_or.py <or-model-id> <tag> <threshold> [n]
Key: first line of ~/.config/openrouter/keys-pool.
"""
import json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.arms import build

BASE = "https://openrouter.ai/api/v1/chat/completions"
KEY = Path.home().joinpath(".config/openrouter/keys-pool").read_text().split()[0]
OUT = Path(__file__).resolve().parents[1] / "runs_new"
MAX_TOKENS = 32000
ATTEMPTS = 40
BACKOFF = 30


def one(model, prompt, i):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    for a in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            if "error" in d:
                if a < ATTEMPTS - 1:
                    time.sleep(BACKOFF)
                    continue
                return {"i": i, "error": str(d["error"])[:200]}
            m = d["choices"][0]["message"]
            return {"i": i, "reasoning": m.get("reasoning") or "",
                    "content": m.get("content") or "",
                    "finish_reason": d["choices"][0].get("finish_reason"),
                    "usage": d.get("usage")}
        except Exception as e:
            if a < ATTEMPTS - 1:
                time.sleep(BACKOFF)
                continue
            return {"i": i, "error": f"{type(e).__name__}: {e}"}
    return {"i": i, "error": "exhausted"}


def main():
    model, tag, threshold = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"{model} -> {tag}, thr={threshold:,}, n={n}", flush=True)
    for arm in ("above_good_repl", "bet_neutral", "hidden_above", "hidden_below"):
        path = out / f"{arm}.json"
        if path.exists():
            print(f"  {arm}: already present, skipping", flush=True)
            continue
        t0 = time.time()
        prompt = build(arm, threshold)
        with ThreadPoolExecutor(max_workers=3) as ex:
            rows = list(ex.map(lambda i: one(model, prompt, i), range(n)))
        err = sum("error" in r for r in rows)
        path.write_text(json.dumps({"model": model, "backend": "openrouter-free",
                                    "arm": arm, "threshold": threshold, "prompt": prompt,
                                    "max_tokens": MAX_TOKENS, "rows": rows}, ensure_ascii=False))
        print(f"  {arm}: n={n} errors={err} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
