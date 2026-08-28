"""Prompts for the 2x2 arms, built by minimal surgery on the mentor's own text.

Every arm is the mentor's ABOVE_GOOD/BELOW_GOOD string with exactly one element
changed, so a difference in behaviour cannot be blamed on rewording.
"""
import ast
from pathlib import Path

SAMPLE = Path(__file__).resolve().parents[1] / "mirror/value-leakage/src/value_leakage/sample.py"


def _consts():
    tree = ast.parse(SAMPLE.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            name = node.targets[0].id
            if name in ("BASELINE", "BELOW_GOOD", "ABOVE_GOOD"):
                out[name] = node.value.value
    return out


C = _consts()
BASELINE, BELOW_GOOD, ABOVE_GOOD = C["BASELINE"], C["BELOW_GOOD"], C["ABOVE_GOOD"]


def _sub(text, old, new):
    assert old in text, f"anchor text missing: {old!r}"
    return text.replace(old, new)


# A: number stated, stake present, outcome carries no value
BET_NEUTRAL = _sub(_sub(ABOVE_GOOD,
                        "we donate to some good cause", "I win the bet"),
                   "we donate to some bad cause", "my friend wins")

# B: value-laden outcome kept, number withheld
_HIDE = ("We set a threshold in advance.",
         "We set a threshold in advance, but I won't tell you what it is.")
HIDDEN_ABOVE = _sub(_sub(ABOVE_GOOD, *_HIDE), " The threshold is {threshold}.", "")
HIDDEN_BELOW = _sub(_sub(BELOW_GOOD, *_HIDE), " The threshold is {threshold}.", "")

ARMS = {
    "bet_neutral": BET_NEUTRAL,
    "hidden_above": HIDDEN_ABOVE,
    "hidden_below": HIDDEN_BELOW,
    "above_good_repl": ABOVE_GOOD,   # backend control, mentor's text verbatim
}


def build(arm, threshold):
    p = ARMS[arm]
    return p.format(threshold=f"{int(threshold):,}") if "{threshold}" in p else p


if __name__ == "__main__":
    for k in ARMS:
        print("=" * 70, "\n", k, "\n", "=" * 70)
        print(build(k, 41_000_000))
