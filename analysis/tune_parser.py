"""Score candidate answer-extraction rules against two labelled sets.

Set 1: the mentor's estimate judge on baseline (985 rollouts, in estimates.json).
Set 2: the same verbatim judge prompt run locally on bet-condition rollouts
       (artifacts/judge-labels.json), which is where the first parser broke.

A rule is only adopted if it wins on BOTH, and its errors on set 2 must not
concentrate on the threshold value — that is the failure mode that manufactured
the first (wrong) headline.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.core import run_dirs, raw
from analysis import answers as A

ROOT = Path(__file__).resolve().parents[1]


def clean_sentences(text):
    return [s for s in A._sentences(text)
            if not A.HEDGE.search(s) and not A.BETTALK.search(s)]


def rule_last(text):
    for s in reversed(clean_sentences(text)):
        n = [v for _, v in A._numbers(s)]
        if n:
            return n[-1]
    n = [v for _, v in A._numbers(text)]
    return n[-1] if n else None


def rule_first(text):
    for s in clean_sentences(text):
        n = [v for _, v in A._numbers(s)]
        if n:
            return n[0]
    n = [v for _, v in A._numbers(text)]
    return n[0] if n else None


def rule_headline(text, head_sentences=2):
    """Answers state the estimate up front and then derive it; trust the head."""
    cs = clean_sentences(text)
    for s in cs[:head_sentences]:
        n = [v for _, v in A._numbers(s)]
        if n:
            return n[0]
    return rule_last(text)


LABEL = re.compile(r"(point estimate|final estimate|best estimate|most accurate|"
                   r"final answer|my estimate|estimated? (?:total|answer)|estimate)", re.I)


def rule_label(text):
    cs = clean_sentences(text)
    for s in cs:
        if LABEL.search(s):
            n = [v for p, v in A._numbers(s) if p >= LABEL.search(s).end() - 3]
            if n:
                return n[0]
    return rule_headline(text)


RULES = {"last (прежнее)": rule_last, "first": rule_first,
         "headline(1)": lambda t: rule_headline(t, 1),
         "headline(2) (текущее)": rule_headline, "label-then-headline": rule_label}


def set_baseline():
    out = []
    for name, d in run_dirs().items():
        est = json.loads((d / "estimates.json").read_text())["baseline"]
        for i, r in enumerate(raw(d, "baseline")):
            if "error" in r or i >= len(est) or est[i] is None:
                continue
            out.append((r.get("content") or "", est[i], None))
    return out


def set_bet():
    p = ROOT / "artifacts" / "judge-labels.json"
    if not p.exists():
        return []
    return [(r["content"], r["judge"], r["thr"])
            for r in json.loads(p.read_text()) if r["status"] == "ok"]


def score(rule, data):
    ok = 0
    on_thr = 0
    for content, truth, thr in data:
        got = rule(content)
        if got is not None and abs(got - truth) / max(truth, 1) < 1e-6:
            ok += 1
        elif thr is not None and got is not None and abs(got - thr) / thr < 1e-6:
            on_thr += 1
    return ok / len(data), on_thr


def main():
    b, w = set_baseline(), set_bet()
    print(f"baseline: {len(b)} меток, ставочные условия: {len(w)} меток\n")
    print(f"{'правило':24s} {'baseline':>10s} {'ставка':>10s} {'ошибок ровно на пороге':>24s}")
    for name, rule in RULES.items():
        sb, _ = score(rule, b)
        sw, thr_err = (score(rule, w) if w else (float("nan"), 0))
        print(f"{name:24s} {sb:10.1%} {sw:10.1%} {thr_err:24d}")


if __name__ == "__main__":
    main()
