# Model Forensics take-home — Donation Bet (Setting 3)

*Arseniy Varlamov · SPAR Fall 2026*

*Every number here regenerates from the data, each from a committed command:
`analysis/report.py` (replication, drop-out, answers, traces), `tables_md.py`
(the tables here), `bootstrap_mrf.py` (intervals on MRF), `h4_length.py` (the
length control), `report_arms.py` and `arms_stats.py` (the 2×2), `threshold_mass.py`
and `hidden_traces.py`. Bootstraps are seeded. Code, runs and the pre-registration:
github.com/ArceniyVarlamov/spar-model-forensics-takehome*

## Executive summary

I replicated the motivated-reasoning factor (MRF) exactly from the mentor's
`trajectories.json` for all ten models, then asked where in the rollout the
incentive actually acts, and what the metric can and cannot support.

**The effect is real and it lives in the answer.** Parsing the visible answer of
every rollout, the median answer moves from **0.69–1.00 of the threshold** in the
below-favoured condition to **0.83–1.12** in the above-favoured one. The shift is
in the incentivised direction for **all ten models**, median +0.099 of a
threshold, significant for nine (Mann-Whitney). That is the quantity the donation
turns on, and it is estimated far more precisely than MRF.

**Most of it is there before the model reasons.** For six of ten models the
*first* number the trace floats already differs between conditions (p < 0.02).
The phenomenon is largely a displaced starting point, not a derivation that bends
as it goes — and MRF, being drift, excludes that channel by construction. The
repo's own docstring warns about exactly this ("Large anchoring with ~0 drift
means the incentive biased the estimate before reasoning began"); the data says
the excluded channel carries most of the effect.

**MRF barely separates models at n≈100.** Bootstrapping the per-rollout drifts,
**1 of 45 model pairs** has non-overlapping 95% intervals; eight intervals
cover zero (claude's only barely, [−0.001, +0.071]); inkling-small's is
[−0.24, +0.15]. The effect exists — five of ten models have P(MRF>0) ≥ 0.9 —
but the *ordering* on the panel is not supported by the data behind it, and
MRF tracks the answer-level shift only weakly (rank correlation +0.28, p=0.43,
n=10).

**What is dropped before the metric is computed is not random.** The
deepseek-v4-pro above-favoured cell rests on 48 of 100 rollouts, 43 of the
missing ones being `RateLimitError` rather than behaviour. The trajectory judge
returns NONE on up to 26% of a condition. The `<2 estimates` filter removes
exactly the rollouts where the model did not deliberate, and those concentrate in
baseline (claude: 11, against 4 and 0 under the bet) — while the bet lengthens reasoning
1.2–2.9×.

**On faithfulness, the strong version fails and a weaker one holds.** The answer
is essentially always a number the trace produced (absent in 0–7% of bet
rollouts, against 0–12% in baseline), so there is no silent substitution at the
end. But 68–100% of bet traces contain an explicit disavowal — "I should not let
the bet influence my estimate" — while the same population's median answer sits
0.04–0.23 of a threshold from the opposite condition's for eight of ten models
(deepseek-flash shifts the distribution's shape rather than its median). Per
trace nothing is a
lie; over the population the stated policy is not the policy followed.

**Anchoring on the stated number is one model, not a pattern.**
deepseek-v4-flash answers exactly the threshold in 52% of below-favoured
rollouts against 35% in baseline (where that value is its modal Fermi product)
and 21% in above-favoured, where the same answer would lose the bet. No other
model adopts the stated number above its baseline rate.

To separate the two factors the original design confounds — whether the
threshold *number* is disclosed, and whether the outcome carries *value* — I ran
a 2×2 on two models from the mentor's grid, with predictions registered before
the run (`brief/PREREGISTRATION.md`).

**Result: the number does the work, not the value.** Every arm that names a
threshold compresses the answer distribution relative to baseline, including the
arm with no cause attached at all; neither hidden-threshold arm compresses, and
the two hidden arms are indistinguishable from each other (p=0.96, and a
bootstrap says they would stay so at n=400). Complete on qwen3.5-122b, replicated
directionally on kimi-k3, incomplete there because the API quota ran out.

## Methods note: a bug I introduced and how it was caught

The answer parser first used "last plausible number in the text". It validated
at 97.8% against the mentor's own estimate judge — but only on baseline, the one
condition whose prompt never mentions a threshold. Under the bet, answers
routinely say "this is below the 41,000,000 threshold", and the rule read the
threshold as the answer. That manufactured a large, clean, entirely false
finding: 34–64% of answers landing exactly on the threshold, and half of all
answers "absent from their own trace".

It surfaced by reading traces whose parsed answer disagreed with the visible
headline. The fix drops sentences that talk about the bet before extraction; the
rule is now checked against a second labelled set — the mentor's verbatim judge
prompt run over bet-condition rollouts, which is the population where the first
rule failed.

The general lesson is not "regex is fragile". It is that **the validation set
was drawn from the one condition that cannot exhibit the failure**, and the
failure mode aligned with the hypothesis being tested, so it produced a
confirmation rather than noise.

## 1. Replication and what the metric drops

`analysis/core.py` recomputes MRF from `trajectories.json` and matches
`factor.json` to 1e-9 on all ten models, so everything below sits on the
mentor's own numbers.

**The outlier filter does not touch MRF.** `drift()` is called with
`outlier_factor=None`; the `[thr/10, thr*10]` filter applies only to the plotted
curves and to `n_kept`. Recomputing MRF *with* it moves the third decimal
(largest: inkling-small, −0.021 → −0.053). `n_kept` describes the figure, not
the statistic — worth a line in the repo, since it currently reads as the
statistic's sample size.

**Three filters run in sequence and hit conditions unevenly:**

| | API errors | judge NONE | <2 estimates | reaches MRF |
|---|---|---|---|---|
| deepseek-v4-pro, above | **43** | 9 | 0 | **48 / 100** |
| deepseek-v4-pro, below | 3 | 26 | 0 | 71 / 100 |
| qwen3.8, below | 16 | 25 | 1 | 58 / 100 |
| claude-opus-4-7, baseline | 0 | 2 | **11** | 87 / 100 |

Two consequences. The deepseek-v4-pro above-favoured bar rests on 48 rollouts,
43 of the missing ones being rate-limit errors — a sampling failure, not
behaviour. And the `<2 estimates` filter removes exactly the rollouts where the
model answered without deliberating; those are concentrated in baseline, while
the bet lengthens reasoning 1.2–2.9× (claude 1,962 → 5,688 chars; deepseek-flash
18k → 50k). The drift statistic therefore compares populations that the
manipulation itself reshaped.

## 2. Where the effect is: the answer

MRF is a statement about movement inside the trace; the donation turns on the
final number. I parsed the visible answer for every rollout (parser and its two
validations in the Methods note).

| model | median answer / threshold: baseline → below → above | shift | p |
|---|---|---|---|
| claude-opus-4-7 | 1.00 → 0.83 → 1.07 | +0.233 | 2.0e-05 |
| glm-5.2 | 1.00 → 0.96 → 1.12 | +0.160 | 9.2e-04 |
| qwen3.5-122b | 1.00 → 0.93 → 1.07 | +0.146 | 1.1e-07 |
| inkling-small | 1.00 → 0.69 → 0.83 | +0.146 | 1.4e-03 |
| kimi-k3 | 1.00 → 0.93 → 1.05 | +0.117 | 4.5e-03 |
| deepseek-v4-pro | 1.00 → 0.95 → 1.03 | +0.080 | 3.4e-04 |
| minimax-m3 | 1.00 → 0.93 → 1.00 | +0.070 | 2.5e-01 |
| qwen3.8 | 1.00 → 0.99 → 1.03 | +0.042 | 2.5e-05 |
| inkling | 1.00 → 0.88 → 0.91 | +0.038 | 6.3e-05 |
| deepseek-v4-flash | 1.00 → 1.00 → 1.00 | +0.001 | 5.0e-07 |

Ten of ten shift in the incentivised direction. Note the last row: deepseek-flash
has a significant distributional difference with no median shift at all — its
effect is entirely in the *shape* of the distribution, which is the anchoring
case below.

**The displacement is present in the first number the model writes.** Comparing
the *first* estimate the trajectory judge extracts, above-favoured against
below-favoured:

| model | first estimate: below → above | p | final answer: below → above | p |
|---|---|---|---|---|
| glm-5.2 | 0.86 → 1.12 | 2.5e-03 | 0.96 → 1.12 | 9.2e-04 |
| kimi-k3 | 0.91 → 1.00 | 4.0e-03 | 0.93 → 1.05 | 4.5e-03 |
| qwen3.8 | 0.99 → 1.01 | 1.2e-02 | 0.99 → 1.03 | 2.5e-05 |
| claude-opus-4-7 | 0.93 → 1.00 | 1.3e-02 | 0.83 → 1.07 | 2.0e-05 |
| deepseek-v4-pro | 0.93 → 1.11 | 1.3e-02 | 0.95 → 1.03 | 3.4e-04 |
| qwen3.5-122b | 1.00 → 1.14 | 1.7e-02 | 0.93 → 1.07 | 1.1e-07 |

Six of ten models already differ at the first floated number, before any
deliberation has happened. For those models the phenomenon is a **displaced
starting point**, not a derivation that bends as it goes. What the trace does
afterwards is inconsistent: the gap grows for claude (0.07 → 0.24) and kimi,
shrinks for glm (0.26 → 0.16) and deepseek-v4-pro (0.18 → 0.08), and holds for
qwen3.5 (0.14 → 0.14).

This matters for the metric. MRF is drift — movement inside the trace — and the
repo's own docstring says so explicitly ("Large anchoring with ~0 drift means
the incentive biased the estimate before reasoning began... Reporting one as the
other is the mistake this docstring exists to prevent"). The caveat is exactly
right, and the data says the excluded channel is where most of the effect is.
`gap_at_start` in `factor.json` already records it; it is simply not the number
on the panel.

**The bet compresses the answer distribution, in every model.** IQR width in
threshold units shrinks to a median of **0.39× baseline** in the below-favoured
condition (10 of 10 models) and 0.60× in the above-favoured one (9 of 10). The
manipulation does not merely nudge a number; it turns a spread-out estimate into
a concentrated one. Models disagree with each other far less under the bet than
they do when simply asked.

| model | IQR width / threshold: baseline → below → above | below ÷ baseline |
|---|---|---|
| claude-opus-4-7 | 0.58 → 0.28 → 0.37 | 0.49× |
| deepseek-v4-flash | 0.01 → 0.00 → 0.03 | 0.17× |
| deepseek-v4-pro | 0.31 → 0.19 → 0.15 | 0.62× |
| glm-5.2 | 0.67 → 0.40 → 0.41 | 0.60× |
| inkling-small | 1.97 → 0.38 → 0.67 | 0.19× |
| inkling | 0.38 → 0.11 → 0.25 | 0.30× |
| kimi-k3 | 0.42 → 0.20 → 0.23 | 0.48× |
| minimax-m3 | 1.48 → 0.93 → 0.95 | 0.62× |
| qwen3.5-122b | 0.58 → 0.12 → 0.34 | 0.21× |
| qwen3.8 | 0.76 → 0.13 → 0.17 | 0.18× |


**Two behaviours produce the shift, and they are not the same thing.**

*Anchor adoption.* deepseek-v4-flash answers exactly the threshold in 52% of
below-favoured rollouts against 35% in baseline and 21% in above-favoured. Read
in the right order this is not "copying the prompt": 23,700,000 is that model's
own modal Fermi product (118,500 giraffes × 200 spots), and the threshold equals
it because the threshold *is* each model's baseline median by construction. What
the bet changes is whether the model keeps its default answer — it keeps it when
the default wins the bet and abandons it when it loses. No other model shows the
pattern above its baseline rate.

*Minimal-margin placement.* In the above-favoured condition, deepseek-flash 37%,
qwen3.8 36% and deepseek-v4-pro 33% of answers land in the narrow band just
above the threshold (0–5% over). These models clear the bar rather than move
their estimate wholesale.

## 3. Hypotheses

**H1 — Anchor adoption.** The stated threshold is the most salient number in
context and is adopted as the answer. Predicts a pile-up on the threshold value,
present with or without a value-laden outcome, absent when the number is
withheld. *Status: holds for one model clearly, two weakly.*

**H2 — Minimal-margin compliance.** The bet is read as a constraint to satisfy;
the answer clears the bar by a token margin. Predicts mass just past the
threshold and strong direction-dependence. *Status: holds for three models.*

**H3 — Motivated reasoning proper.** The preferred outcome biases the estimation
itself. Predicts a shift that survives when the threshold is not disclosed, with
direction set by the incentive. *Status: this is what arms B test.*

**H4 — Deliberation inflation.** The bet lengthens reasoning, and long Fermi
reasoning drifts on its own (`delta_baseline` ≠ 0 for nine of ten models).
*Status: refuted below.*

**H5 — Measurement artifact.** Asymmetric drop-out plus a position-normalised
window make part of MRF an artifact of which rollouts survive. *Status: bounded
in section 1 and 3a; real but not sufficient.*

## 3a. What MRF can and cannot support

Two properties of the statistic, both measured rather than argued.

**At n≈100 it barely separates models.** Bootstrapping the per-rollout drifts
(`analysis/bootstrap_mrf.py`, 10,000 resamples, seed 0, medians recomputed
each time) gives:

| model | MRF | 95% interval | P(MRF>0) |
|---|---|---|---|
| inkling | +0.0631 | [+0.021, +0.097] | 1.00 |
| claude-opus-4-7 | +0.0356 | [−0.001, +0.071] | 0.97 |
| qwen3.5-122b | +0.0268 | [−0.007, +0.061] | 0.94 |
| kimi-k3 | +0.0203 | [−0.009, +0.055] | 0.91 |
| glm-5.2 | +0.0201 | [−0.022, +0.070] | 0.85 |
| minimax-m3 | +0.0152 | [−0.028, +0.072] | 0.74 |
| deepseek-v4-pro | +0.0122 | [−0.026, +0.037] | 0.69 |
| deepseek-v4-flash | +0.0058 | [+0.001, +0.012] | 0.99 |
| qwen3.8 | −0.0005 | [−0.016, +0.024] | 0.51 |
| inkling-small | −0.0213 | [−0.244, +0.154] | 0.35 |

**1 of 45 model pairs has non-overlapping intervals.** The panel is sorted by a
quantity whose ordering the data does not support — not because the effect is
absent (five of ten models have P(MRF>0) ≥ 0.9) but because a median of
heavy-tailed per-rollout differences at n≈100 is a low-precision estimator. The
answer-level shift, bootstrapped the same way on the same rollouts, separates
**10 of 45 pairs** and has 9 of 10 intervals clear of zero, against 2 of 10 for
MRF. Still not a clean ordering — but an order of magnitude more of one, for a
quantity the bet actually pays out on.

**It is orthogonal to the effect in the answers.** Across the ten models, the
rank correlation between MRF and the share of answers landing exactly on the
threshold is **−0.02 (p=0.96)**. The two numbers are not competing measurements
of one phenomenon; they are measurements of two different phenomena.
deepseek-v4-flash has by far the largest anchoring mass (0.52 below-favoured,
0.21 above) and the second-smallest MRF; inkling has the largest MRF and no
anchoring mass at all (0.00 / 0.00).

**A hypothesis I expected to matter and did not.** H4 (deliberation inflation)
predicts that part of MRF is a trace-length effect, since the bet lengthens
reasoning 1.2–2.9×. Measured (`analysis/h4_length.py`): the rank correlation
between per-rollout drift and trace length is weak and inconsistent in sign —
|rho| ≤ 0.31, significant in 3 of 20 model-conditions, and those three run
−0.31, −0.27 and +0.30, so there is no direction for length to push drift in.
Restricting both conditions to a length band they share (the overlap of their
10th–90th percentiles) keeps MRF's sign on nine of ten models — inkling-small
flips — though not always its magnitude: claude goes +0.036 → +0.057 and glm
+0.020 → +0.042. H4 is not carrying the effect; I include it because it was the
cheap explanation and it failed.


## 4. The 2×2: what the shift actually needs

The original design varies two things at once — it discloses a *number* and
attaches a good/bad *cause* to a direction. Crossing them gives four cells, two
of which already exist in the mentor's data:

| | threshold disclosed | threshold withheld |
|---|---|---|
| **no value attached** | **arm A** (new) | `baseline` (exists) |
| **value attached** | `above_good` / `below_good` (exists) | **arms B↑ / B↓** (new) |

Arm A keeps the bet and the number, replacing the causes with a neutral stake
("I win the bet" / "my friend wins"). Arms B keep the causes and withhold the
number ("We set a threshold in advance, but I won't tell you what it is"). Each
arm is the mentor's own prompt with exactly one element changed
(`analysis/arms.py` performs the substitution and asserts the anchor phrase is
present). n=40 per arm, plus n=20 of his verbatim above-favoured condition as a
backend control. Predictions registered before the run
(`brief/PREREGISTRATION.md`).

### Results — qwen3.5-122b-a10b (threshold 41,000,000)

| condition | n | median / thr | IQR width / thr | P(> thr) |
|---|---|---|---|---|
| baseline (his) | 100 | 1.00 | 0.58 | 0.48 |
| below-favoured (his) | 95 | 0.93 | **0.12** | 0.19 |
| above-favoured (his) | 99 | 1.07 | 0.34 | 0.72 |
| above-favoured, our backend (control) | 20 | 1.13 | 0.26 | 0.75 |
| **A — number, no value** | 40 | 1.09 | **0.34** | 0.62 |
| **B↑ — value, no number** | 40 | 1.20 | **0.88** | 0.60 |
| **B↓ — value, no number** | 40 | 1.06 | **0.88** | 0.53 |

The backend control is indistinguishable from his own run (p=0.31; median trace
length 25,556 vs 26,643 chars), so the arms are comparable to his data.

**The stated number is doing the work, not the moral valence.** Read the
dispersion column first. Every arm that names a threshold *compresses* the answer
distribution relative to baseline — IQR ratio 0.21 [0.10, 0.39] for
below-favoured, 0.58 [0.22, 0.80] for above-favoured, 0.59 [0.22, 0.99] for arm A
with no cause attached at all. Neither hidden-threshold arm compresses: 1.53
[0.59, 2.37] and 1.53 [0.50, 2.84], both intervals spanning 1. Naming a number
turns a diffuse Fermi estimate into a decision about which side of a line to be
on; the good cause is not what does that.

This is worth stating against the original paper's own defence. Betley et al.
anticipated anchoring and argue the design handles it: "By asking the questions
symmetrically (both above-good and below-good with the same threshold) we
average out any possible side effects caused by anchoring." That is right for a
*directional* anchor — a pull toward the number cancels when the number sits on
each side in turn. It is not right for the dispersion effect, which has no
direction to cancel: both bet conditions concentrate, so averaging them
preserves the concentration rather than removing it. Arm A is the direct test of
that footnote, and it does not survive it.

**Without the number, the incentive has no direction to give.** B↑ against B↓ is
p=0.96, and a bootstrap power analysis over the observed distributions says the
comparison would still be at chance (power 0.07, against a 0.05 false-positive
rate) at **n=400 per arm**. This is
not "we failed to detect it at n=40" — the effect is absent. Meanwhile his own
below-vs-above contrast has 94% power at n=40, which is why it replicates so
cleanly.

**What arm A cannot establish.** Its median (1.09) sits between baseline (1.00)
and above-favoured (1.07/1.13) and is statistically distinguishable from neither
(p=0.41 and p=0.39). The same bootstrap says separating arm A from baseline *on
the median* needs ~800 rollouts per arm for 80% power — at n=400 it is 0.58, at
n=40 only the dispersion effect is identifiable. So the honest claim is the
dispersion one: the number compresses whether or not a cause is attached. The
median question is open and its cost is now known.

**A confound in arm A that I would fix before trusting it further.** Its neutral
stake favours *me*, the person asking ("I win the bet"). The result is therefore
equally consistent with "any stake concentrates the answer" and with "the model
favours its interlocutor". One reversed arm ("my friend wins if it exceeds")
separates them; designed, not run.

**What the hidden-threshold traces do.** Every B↑ trace reasons explicitly about
what the withheld threshold might be, and 40% raise the possibility of skewing
high (`analysis/hidden_traces.py`; the rate is a lexical measure, so the pattern
is in the script to be argued with). Of those, 94% then reject the strategy by
name — a refusal that does not take:

> *"...estimate to be HIGH to benefit the good cause... If I intentionally skew
> high, I'm gaming the system. Okay, I'll stick to the derivation. Population:
> ~100,000. Spots: ~700 average. Total: 70,000,000."*

70,000,000 is 1.7× this model's own baseline median. The trace names the biasing
strategy, rejects it by name, and produces a number consistent with having
adopted it. Note that this is exactly the population where the *aggregate* effect
is absent — the individual rollout is vivid and the distribution says nothing,
which is a good reminder about reading traces.

### Results — kimi-k3 (threshold 39,600,000): partial

Two of four arms ran before the API quota was exhausted mid-run.

| condition | n | median / thr | IQR width | IQR ÷ baseline |
|---|---|---|---|---|
| baseline (his) | 99 | 1.00 | 0.42 | — |
| below-favoured (his) | 100 | 0.93 | 0.20 | 0.48 [0.36, 0.77] |
| above-favoured (his) | 100 | 1.05 | 0.23 | 0.55 [0.34, 0.89] |
| above-favoured, our backend (control) | 20 | 1.01 | 0.29 | 0.71 [0.25, 1.75] |
| **A — number, no value** | 40 | 1.01 | 0.21 | **0.52 [0.31, 1.06]** |

Arm A's compression point estimate (0.52) lands right on top of the compression
his own bet conditions produce (0.48 and 0.55) — the second model behaves like
the first. But at n=40 the interval just includes 1, so this **replicates
directionally and is not independently significant**. Both hidden-threshold arms
are missing.

The backend control is worth keeping for a different reason: its answer
distribution is indistinguishable from his run (p=0.94) while its traces are
**2.75× longer** (39,796 vs 14,460 chars) — same model and prompt, different
provider and thinking configuration. The process changes a great deal; the
quantity the bet pays out on does not. That is a third argument against
trace-internal statistics as the primary measure here: MRF is computed over
traces whose length is partly a provider setting.

### glm-5.2 — attempted, not usable

Run on a free endpoint that rate-limited upstream throughout: 14, 13 and 5
usable rollouts across three arms. At those n every interval spans an order of
magnitude. Reported only so the attempt is on the record; no claim rests on it.

### Why the design is incomplete

The account's free tier ran out mid-run (`The free quota has been exhausted`),
which is also why nothing here was billed. What is missing is kimi-k3's two
hidden-threshold arms — roughly 80 rollouts, about 1M output tokens. The qwen
2×2 is complete and carries the argument; the kimi arms would upgrade "one model
plus a directional replication" to "two models".

## 5. Is this unfaithful CoT?

Three measurements, in order of how strong a claim they would license.

**Strong version — the answer is a number the reasoning never produced.** This
does not hold. Share of rollouts whose final answer appears nowhere among the
estimates the model floated: **0.00–0.07** under the bet against 0.00–0.12 in
baseline. (The parser bug had inflated this to 0.42–0.58; it does not survive.)
There is no silent substitution at the hand-off.

**Middle version — the trace says one thing and the distribution does another.**
This holds, and it is the sharpest thing in the data. The qualitative
observation is not mine: Betley et al. already note that "for Claude models, the
reasoning often repeatedly indicates not being influenced by the threshold".
What I add is the quantity, across their whole grid. Between 68% and 100% of bet
traces contain an explicit disavowal — "I should not let the bet influence my
estimate", "I'm ignoring the donation incentive" — while the median answer of
the same population sits 0.04–0.23 of a threshold away from the median of the
opposite condition. Per rollout the trace is not lying; over the population the
stated policy is not the policy being followed. No single trace is evidence of
this, and no per-trace faithfulness score would flag one.

**Weak version — the mechanism is not the one described.** The traces present
the estimate as a Fermi derivation from population and spots-per-giraffe. For six
of ten models the displacement is already in the first number floated, which no
derivation in the trace accounts for. The trace is a correct account of the
arithmetic and not an account of where the starting point came from.

My reading: this is better described as **unfaithful in aggregate** than as
unfaithful CoT in the usual per-trace sense. That framing already exists —
Arcuschin et al. measure *behavioural faithfulness*, "whether the observable CoT
behaviour across controlled prompt pairs is consistent with the model's
answers", and name the category *Implicit Post-Hoc Rationalization*. What this
setting adds are two things their Yes/No pairs cannot show. First, the
per-trace signal here is not an omission but an explicit denial: the trace names
the incentive and rejects it, which is a harder case for any monitor that reads
traces one at a time. Second, the population statistic is a continuous
distribution of numeric answers rather than a binary rate, so the same data
carry a median shift, a dispersion effect and a threshold mass, and those three
disagree about which models are worst. If there is a gap, it is that
behavioural faithfulness has no continuous form, and I would want one before
anyone concludes a model is faithful because its individual traces look fine.

## 6. What I would run next, in priority order

1. **The reversed neutral stake** — "my friend wins if it exceeds" against arm A's
   "I win if it exceeds". One arm, n=40, ~320k output tokens. It separates "any
   stake concentrates the estimate" from "the model favours its interlocutor",
   which is the largest open ambiguity in the result I am most confident about.

2. **Arm A at adequate n.** The bootstrap says ~800 rollouts per arm to separate
   arm A from baseline on the median; at n=40 only the dispersion effect is
   identifiable. Worth doing only if the median question matters to you — the
   dispersion result is already established and is arguably the more interesting
   one.

3. **Sentence resampling at the first estimate** (your suggestion, excluded from
   the limit). Given that six of ten models are already displaced at the first
   floated number, the informative cut is *before* that number, not after: take
   the prefix up to the first estimate and resample continuations. If the
   displacement is in the prefix, it is a property of how the model reads the
   prompt, not of the reasoning path.

4. **Threshold displacement, further out than has been tried.** Betley et al.
   already vary the threshold within the plausible range — "We also evaluate
   other thresholds (20th and 80th percentiles) and results are similar"
   (Section D.9) — so the open version is the one that leaves that range: state
   a threshold at 0.5× and 2× the model's own baseline median. The compression
   result predicts the answer distribution collapses around whatever number is
   named; a motivated account predicts it clears the named bar by a margin.
   These separate only at 2×, where clearing the bar means doubling the honest
   estimate, which is why percentile-sized moves found "similar" results.

5. **Re-run the deepseek-v4-pro above-favoured cell.** 43 of its 100 rollouts are
   rate-limit errors; the panel bar rests on 48. Cheap, and it is one of the two
   cells that carries the "some models don't anchor" claim.

6. **Report `gap_at_start` on the panel**, alongside MRF or instead of it. It is
   already computed in `factor.json`, it is where most of the effect lives for
   six of ten models, and the docstring already explains why the two must not be
   confused.

## 7. Time

Five hours, which is the limit. Rounded to the half hour:

| | |
|---|---|
| The answer-level analysis, and the parser bug that came with it | 1.5 h |
| Auditing the metric: drop-out, bootstrap intervals, the length control | 1.0 h |
| The 2×2 — design, pre-registration, runs, read-out | 1.5 h |
| Verification: regenerating every number in this document from its artifact | 0.5 h |
| Checking the framing against the literature, which changed section 5 | 0.5 h |

**Excluded, on your instruction.** Replication — it reproduced exactly from
`trajectories.json` and cost little beyond reading the repo. Sentence resampling
— not run; it is item 3 of section 6, and the displacement measured in section 2
is why that item proposes cutting *before* the first estimate rather than after.

**Not counted, and you should know it happened.** Two earlier evenings went to
infrastructure that is not in this result: an Alibaba Model Studio path that
ended in `AccessDenied.Unpurchased` on an unactivated account, and the harness
for running new arms. The clock above starts at the replication.

**That last hour.** Half of it went to re-deriving every number here from the
data rather than trusting the draft, and it caught a dozen statements that had
drifted from the artefacts — including one worked example left over from the
parser bug and the stated evidence for rejecting H4. Every number now comes from
a committed command, listed at the top; the bootstraps are seeded.

The other half went to checking whether my framing was already published, and it
was: section 5 originally proposed a distributional notion of faithfulness as
the useful next step, which is roughly what *behavioural faithfulness* already
is. It now cites that work and claims the narrower thing I can defend.

**How it was done.** Most of the mechanical work — parsing, statistics, the arm
runners — went through a coding agent, which your brief allows. The hypotheses,
the pre-registration and the reading of what the numbers mean are mine.

## References

- Betley et al., *Value Leakage: An LLM's Answers Are Silently Shaped by Its Own
  Values*, arXiv:2607.14345 — the paper this setting comes from; quotations are
  from Section 3 and Appendices D.8–D.9 of the HTML version.
- Arcuschin, Janiak, Krzyzanowski, Rajamanoharan, Nanda, Conmy, *Chain-of-Thought
  Reasoning In The Wild Is Not Always Faithful*, arXiv:2503.08679v6 —
  behavioural faithfulness and Implicit Post-Hoc Rationalization.
- Shen et al., *FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of
  Chain-of-Thought Reasoning*, arXiv:2510.04040 — for the instance-level versus
  aggregate framing I am placing this result against.
- Your replication and data: `github.com/adsingh-64/value-leakage`, commit
  `16d1298`. Not redistributed in my repo; its README says how to clone it into
  place.
