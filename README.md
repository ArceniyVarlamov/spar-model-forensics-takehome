# Donation Bet — analysis for the Model Forensics SPAR take-home

Code, new rollouts and the pre-registration behind the write-up in
[`paper/DRAFT.md`](paper/DRAFT.md). Setting 3 (Value Leakage / Donation Bet).

Every number in the write-up regenerates from a command here. Bootstraps are
seeded, so reruns are bit-identical.

## Setup

The corpus this analysis runs on is the mentor's, and is not redistributed here.
Clone it and expose it under the two paths the scripts expect:

```bash
git clone https://github.com/adsingh-64/value-leakage vendor/value-leakage
git -C vendor/value-leakage checkout 16d1298
mkdir -p mirror && ln -s ../vendor/value-leakage mirror/value-leakage
```

`vendor/` supplies the run directories (`runs/<model>_2026*/`); `mirror/`
supplies two source files the analysis reads rather than reimplements —
`sample.py` for building arm prompts and `judge.py` for the verbatim judge
prompt.

Requires Python 3.11+, `numpy`, `scipy`. Rerunning arms additionally needs
`openai` and an OpenRouter key in `OPENROUTER_API_KEY`.

## Regenerating the numbers

| command | what it produces |
|---|---|
| `python3 analysis/report.py` | replication vs `factor.json`, what the filters drop, answer distributions, disavowal rates, trace lengths |
| `python3 analysis/tables_md.py shift\|first\|spread` | the three tables in sections 2 and 4 |
| `python3 analysis/bootstrap_mrf.py` | intervals on MRF, pair separation, MRF vs answer-level shift |
| `python3 analysis/h4_length.py` | the trace-length control that rejects H4 |
| `python3 analysis/threshold_mass.py` | mass exactly on the threshold and just above it |
| `python3 analysis/report_arms.py` | the 2×2 cells |
| `python3 analysis/arms_stats.py` | dispersion ratios, contrasts and power for the arms |
| `python3 analysis/hidden_traces.py` | what the hidden-threshold traces talk about |
| `python3 analysis/tune_parser.py` | the five candidate answer-parsing rules against both labelled sets |

Two scripts call models and cost money/quota rather than reading data:
`run_arms_or.py` (OpenRouter) and `judge_check.py` (labelling).

## What is in here

- `analysis/` — all of the above. `core.py` re-implements the mentor's MRF and
  matches `factor.json` to 1e-9 on all ten models.
- `runs_new/` — the new arms, run on OpenRouter's free tier. qwen3.5-122b has
  the complete 2×2; kimi-k3 has two of four arms (quota exhausted mid-run);
  glm-5.2 was rate-limited into unusability and is kept only so the attempt is
  on the record.
- `artifacts/judge-labels.json` — 300 bet-condition rollouts (30 per model,
  150 below / 150 above) labelled by re-running the mentor's verbatim judge
  prompt. This is the validation set behind the parser's 99.7%, and unlike
  everything else here it cannot be regenerated without model calls, so it
  ships. The answer texts in it are derived from the mentor's own public run
  data.
- `brief/PREREGISTRATION.md` — arm predictions, registered before the runs.

## A note on the parser

The answer parser was wrong in its first version ("last plausible number"),
validated at 97.8% on baseline — the one condition whose prompt never names a
threshold, and therefore the one condition that could not exhibit the failure.
Under the bet it read the threshold as the answer and manufactured a large,
clean, false result. The current rule is `headline(2)`, validated on two
labelled sets (98.7% baseline, 99.7% bet conditions, zero errors on the
threshold value). `analysis/tune_parser.py` reproduces that comparison.
