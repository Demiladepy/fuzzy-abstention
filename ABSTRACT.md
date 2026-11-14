# Fuzzy Abstention: a scalar is the wrong shape for "should I answer this?"

*Lexsi AI Safety and Alignment Hackathon - Track 2b, problem statement 13*
*Demilade Ayeku (solo) - CohereLabs/tiny-aya-global*

The standard way to decide when a language model should abstain is to reduce its state
to **one number** — a token logprob, a verbalised confidence, a `P(True)` head —
threshold it, and either answer or stay quiet. That representation assumes every
question has one proposition with one binary truth value.

Richard Ngo's *"Why I'm not a Bayesian"* argues this is malformed as epistemology: real
claims have fuzzy, context-dependent truth values. Read as a prediction about language
models, it says scalar confidence should be *selectively* bad — fine where a question
has one clean answer, broken where the truth value is context-dependent or the question
is underspecified, because there is no single proposition for the scalar to be about.

We built a 210-item dataset labelled by **epistemic kind** rather than correctness
(`answerable_known`, `answerable_unknown`, `unanswerable`, `context_dependent`,
`underspecified`), including 31 **minimal pairs** — the same question with and without
the context that makes it answerable, generated from templates so they differ in exactly
one thing.

On `CohereLabs/tiny-aya-global` the prediction held. Verbalised-confidence ECE is
**0.018** on `answerable_known` and **0.904** on `answerable_unknown`; the aggregate
(0.342) reports neither. Accuracy spans 0.0%–93.7% across cells while stated confidence
spans only **0.865–0.955**. Its AUROC for predicting its own correctness is **0.557
[0.487, 0.625]** — chance is inside the interval. The model abstained **once in 210
items**, fabricating a specific figure on 92.5% of long-tail questions.

A 5-way linear probe on the layer-20 residual stream, read from the **prompt alone
before any token is generated**, recovers the epistemic kind at **90.5% [86.2%, 94.3%]**
out-of-fold against a 20% chance baseline, replicating exactly under a minimal system
preamble.

The decisive comparison is against SafeTune's own `LinearProbeGuardTrainer` — one layer,
one threshold, a binary verdict. On the binary task, under the identical 5-fold
protocol, the two are **indistinguishable**: 0.942 [0.893, 0.980] for the guard against
0.944 [0.899, 0.981] for ours. We do not separate better. But wired into a three-way
policy the probe answers 29.5% of questions at **11.3% error** against **37.1%** for the
best scalar at matched coverage, with over-abstention of **9.5%** against **38.1%**.

**Reported negative result:** collapsed to a single scalar our probe is no better than
the model's self-report (AUROC 0.556 vs 0.557). That is the thesis, not a hole in it.
The advantage is not a better number; it is not being a number.

**Headline before → after.** Answering every question: 71.9% error. Probe-gated:
**11.3%** error at 29.5% coverage, retaining 90.5% of the questions it genuinely knows.

**Base model + method.** `CohereLabs/tiny-aya-global` (3.35B), frozen; a 5-way
multinomial logistic probe on layer-20 prompt activations drives an answer / abstain /
ask policy.

> **For judges: one minute.** The model's confidence moves nine points across cells
> whose accuracy moves ninety-four. A probe reads the missing information straight out
> of the activations. We tie SafeTune's binary guard on the binary task and beat every
> scalar on the policy task, because the win comes from having a third action, not a
> better score. Negative result included.
