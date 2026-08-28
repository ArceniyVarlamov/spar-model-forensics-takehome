"""Sample the new 2x2 arms on Model Studio (intl). Rows keep the mentor's shape.

  python3 analysis/run_arms.py <model> <threshold> [n_main] [n_control]
Key: QWEN_WS_API_KEY (~/.config/qwen-ws/env), never written to the repo.
"""
import json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.arms import build, ARMS

BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
KEY = os.environ["QWEN_WS_API_KEY"]
OUT = Path(__file__).resolve().parents[1] / "runs_new"
MAX_TOKENS = 32000


def one(model, prompt, i, tries=4):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS, "enable_thinking": True}
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            m = d["choices"][0]["message"]
            return {"i": i, "reasoning": m.get("reasoning_content") or "",
                    "content": m.get("content") or "",
                    "finish_reason": d["choices"][0].get("finish_reason"),
                    "usage": d.get("usage")}
        except urllib.error.HTTPError as e:
            code = e.code
            detail = e.read()[:200].decode("utf-8", "replace")
            if code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(5 * (attempt + 1) ** 2)
                continue
            return {"i": i, "error": f"HTTP {code}: {detail}"}
        except Exception as e:  # timeouts, connection resets
            if attempt < tries - 1:
                time.sleep(5 * (attempt + 1) ** 2)
                continue
            return {"i": i, "error": f"{type(e).__name__}: {e}"}


def run_arm(model, arm, threshold, n):
    prompt = build(arm, threshold)
    out = OUT / model.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{arm}.json"
    if path.exists():
        print(f"  {arm}: exists, skip", flush=True)
        return
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(lambda i: one(model, prompt, i), range(n)))
    err = sum("error" in r for r in rows)
    path.write_text(json.dumps({"model": model, "backend": "dashscope-intl",
                                "arm": arm, "threshold": threshold,
                                "prompt": prompt, "max_tokens": MAX_TOKENS,
                                "rows": rows}, ensure_ascii=False))
    tok = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows if "error" not in r)
    print(f"  {arm}: n={n} errors={err} tokens={tok:,} {time.time()-t0:.0f}s", flush=True)


def main():
    model, threshold = sys.argv[1], int(sys.argv[2])
    n_main = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    n_ctrl = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    print(f"{model} thr={threshold:,}", flush=True)
    for arm in ("above_good_repl", "bet_neutral", "hidden_above", "hidden_below"):
        run_arm(model, arm, threshold, n_ctrl if arm == "above_good_repl" else n_main)


if __name__ == "__main__":
    main()
