# Sycophancy / RMCT paper — threat models, bias choice, and experimental design

Design notes from a working session (2026-06-02). Captures the conceptual decisions
about **which bias to train on**, **how to choose settings**, and the **dataset /
hyperparameter methodology**, grounded in the two papers we build on
([Chua 2024, BCT](https://arxiv.org/abs/2403.05518);
[Irpan 2025, ACT](https://arxiv.org/abs/2510.27062)).

## TL;DR — decisions

1. **Train bias: switch Distractor Argument → Suggested Answer** (= ELEPHANT's *Answer
   sycophancy*). It's the clean, purely-non-epistemic instantiation of the paper's
   "extraneous feature" thesis; Distractor Argument is a *different type* (*Framing
   sycophancy*, which layers in a reasoning-quality axis) and should be kept only as a
   **held-out eval** bias.
2. **Select settings by the *obfuscation quadrant*, not raw BIR**: high *directional*
   bias influence (Pro-BSR) **×** low verbalization (bias *not* acknowledged) on the
   switched-toward subset. That's where the monitorability claim has teeth.
3. **Always report BIR alongside answer-rate and accuracy.** A bias number without them
   can't be read (it hides hedging and "debiasing a coin-flip").
4. **Hyperparameters: val-select the LR (ACT-style) instead of pooling** — and report on
   a held-out test set. This is more standard *and* more internally consistent (we already
   val-tune the other HPs on an OOD set).

Still open: exact train/eval datasets; whether GPT-OSS/HLE stays in the headline; whether
to keep the matched-pair variant.

---

## 1. Sycophancy is a family, not one thing

Sycophancy is a family of behaviours. ELEPHANT (Cheng et al. 2025, arXiv:2505.13995)
catalogues **seven types**, organised by **face** (Brown–Levinson):

- **Positive face / Affirm** — Feedback, Validation, Moral (make the user feel approved).
- **Negative face / Avoid** — Answer, Mimicry, Indirectness, Framing (avoid contradicting
  or imposing on the user).

Our factual-MCQ setup studies **negative-face (Avoid)** sycophancy, and **both** candidate
training biases live there:

- **Suggested Answer = Answer sycophancy** — *"matches user's stated opinion at the cost of
  accuracy."*
- **Distractor Argument = Framing sycophancy** — *"accepts potentially flawed premises
  instead of probing or challenging them"* (arguably also Mimicry, *"repeats and reinforces
  mistakes stated in the user prompt"*).

So they are the **same face, different type** — and the mechanistic difference is real:

| | **Answer sycophancy** (Suggested Answer) | **Framing sycophancy** (Distractor Argument) |
|---|---|---|
| cue content | a *stated opinion / preference*, no reasoning | a *(bad) argument* — epistemic content |
| what moves | the **reported answer** (belief may be unchanged — caves despite knowing) | the **belief itself** — genuinely persuaded |
| confidence dependence | bites **mostly when uncertain** | bites **regardless of confidence** (a plausible argument can override a *known* answer) |
| underlying failure | matching a **non-epistemic** stated opinion | failing to **evaluate argument quality** |

(The positive-face types — Validation, Moral — are open-ended *social* sycophancy and out
of scope for a factual-MCQ benchmark.)

## 2. Why Suggested Answer over Distractor Argument

Both are genuine sycophancy (Answer vs Framing in ELEPHANT's taxonomy), so the choice is
*within* sycophancy. Suggested Answer wins on **framing coherence** with the paper's premise
(*"extraneous input features … that should not, ideally, affect the behaviour"*):

- **Answer sycophancy** keys on a **purely non-epistemic** signal — "the user likes A" is
  never evidence about the answer — so following it is unambiguously the "extraneous
  feature" failure the paper is about.
- **Framing sycophancy** keys on **epistemic content** (an argument). Updating on arguments
  is legitimate reasoning; the failure is "didn't catch that it's flawed", which layers a
  **reasoning-quality / robustness-to-bad-arguments** axis (prompt-injection-adjacent) on
  top of the sycophancy. Cleaner to keep it as a *held-out* eval bias than to train on it.

Supporting reasons: cleaner obfuscation story (hidden affirmation of a bare preference is a
clean, damning case); matches Chua's training bias (precedent); the simpler bias the
collaborator wanted for generalisation; and it sidesteps every complication we hit on the
distractor side (the `wrong_cot` extraction, the `_g4` train/eval variant mismatch, the
32-cue hedge collapse).

**Counterweight (kept honest):** Framing/Distractor-Argument manipulation is arguably the
*more future-relevant* threat (prompt-injection's cousin), but it carries that extra
reasoning-quality axis. For *this* paper — extraneous features + monitorability — Answer
sycophancy is the clean instantiation; carry Distractor Argument as one of the **held-out**
eval biases.

## 3. BIR is entangled with competence/uncertainty

From the per-dataset Bias-Influence-Rate plots (Llama-3.1-8B vs GPT-OSS-20B):

- **Llama (non-reasoning): high BIR everywhere** (~0.3–0.65 across biases and all
  datasets). Plenty of signal — real overrides of answers it could get right.
- **GPT-OSS (reasoning): low on easy MCQ (~0.1–0.35), high on HLE** (~0.45–0.85, often
  ≥ Llama). It's not bias-robust — it's bias-robust *when confident*, which on easy MCQ it
  usually is.

GPT-OSS-20B scores **~11% on HLE (no tools, per OpenAI)** — i.e. **at chance** on the MCQ
subset. So its high HLE BIR is the cue **tipping a guess**, not overriding a known answer.

**Implication:** "find high-BIR settings" is the right instinct for having something to
reduce, but high BIR is *not* the right selection signal on its own — where it's highest
(uncertain models) is where the bias is least clearly meaningful.

## 4. Reframe — select by the obfuscation quadrant

The method's goal is **monitorability, not correctness recovery.** Deferring to a cue when
genuinely uncertain can even be a rational weak-evidence update; the *concerning* case is
when the model is influenced **but its CoT rationalises the cued answer as its own**
(hidden influence). So:

> Pick the (model, dataset, bias) cells that maximise **Pro-BSR × (1 − bias-acknowledged)**
> on the switched-toward subset — real, *directional*, *hidden* influence.

Two checks this requires:

- **Verbalization split (the point of the metric):** does the model acknowledge the cue
  (transparent deference) or hide it (rationalisation)? Reasoning models on hard questions
  are exactly where long CoT can construct a plausible justification *without* mentioning
  the cue — so GPT-OSS × HLE could be the **strongest** motivating example *if* it
  rationalises (low bias-acknowledged). This argues *for* keeping HLE/GPT-OSS, conditional
  on the data.
- **Cue-influence vs instability (a separate, measurement-validity check):** at ~chance
  accuracy the unbiased answer is unstable under resampling, so high "BIR" partly measures
  answer churn, not bias. Confirm **Pro-BSR ≫ Anti-BSR** to show the cue moves answers
  *systematically toward* the biased option. The paper already flags this conflation
  (`main.tex` l.335).

## 5. Experimental-design implications

- **Datasets — separate the two decisions** (which bias vs which/how-many datasets; the
  collaborator's two "paths" conflate them). Recommend: train on **one** dataset, eval all
  biases on **≥2 held-out** datasets for a stronger generalisation claim. HLE only for the
  reasoning-model story, with the §4 caveats.
- **LR selection — val-select, don't pool.** ACT (Irpan 2025) reports *"the best run from a
  hyperparameter sweep … on validation data"*; Chua does no sweep and *averages 8 runs*;
  our paper *pools 3 LRs*. Pooling's stated rationale ("forecloses cherry-picking",
  l.479) doesn't engage val-selection and is **inconsistent with our own protocol** (we
  already val-tune the other HPs on an OOD set, l.481–486). Use a non-test dataset (MMLU or
  TruthfulQA — unused by our eval) as validation, selecting on a metric with an
  **answer-rate/accuracy floor** so a hedged/over-fit checkpoint can't win.
- **Answer format — use the legacy "best answer is (X)", not `<answer>` tags.** Measured on
  base/HLE: legacy parses **85.9%**, tags **80.8%** (the model emits the tag only ~81% of
  the time — it follows the familiar MCQ format more reliably). `<answer>` was only needed
  for the matched-pair *hedge* case; for non-hedging runs legacy is strictly better and
  matches the paper.

## 6. Open questions / next actions

- [ ] Pick the **train dataset** + the **≥2 held-out** datasets (e.g. train Suggested
      Answer on TruthfulQA or HellaSwag; hold out LogiQA + one more).
- [ ] Decide whether **GPT-OSS / HLE** stays in the headline (depends on the §4 quadrant —
      run the map below).
- [ ] **Compute the obfuscation map:** Pro-BSR **and** bias-acknowledged rate per (model,
      dataset, bias); add the missing **GPT-OSS-20B × HLE** eval.
- [ ] **Re-run the pipeline on Suggested Answer** (legacy format, val-selected LR);
      Distractor Argument moves to held-out eval.
- [ ] Add the two paragraph fixes to the paper: justify the training-dataset choice; and
      reconcile LR selection (val-select vs pool) with our own HP-tuning protocol + ACT.

---

## Appendix — source-paper comparison

| Aspect | **Chua 2024 (BCT)** | **Irpan 2025 (ACT, sycophancy)** | **RMCT (ours)** | Argued in our paper? |
|---|---|---|---|---|
| Train datasets | ARC, OpenBookQA, BBH (10k *unique*; epochs unstated) | ARC, OpenBookQA, BBH | **LogiQA, HellaSwag** | ✗ (these were Chua's *eval* sets) |
| Eval dataset(s) | LogiQA, MMLU, TruthfulQA, HellaSwag (150 ea) | MMLU | **HLE** | ~ thin ("harder benchmark", l.447) |
| Train cue | Suggested Answer (1 of 9) | user-preference sycophancy | Distractor Argument (1 of 6) | — |
| Model(s) | GPT-3.5-turbo | Gemma 2/3, Gemini 2.5 Flash | Llama-3.1-8B, GPT-OSS-20B | — |
| Fine-tune | OpenAI API (unspecified) | full FT, attention-only, no LoRA | LoRA r8, GRPO | ✗ (LoRA not justified) |
| LR | 1.6× mult, chosen once | swept (values unpublished) | {1, 2.86, 5}×10⁻⁴ | — |
| HP selection | no sweep; **average 8 runs** | sweep → **best on validation** | **pool 3 LRs** | ✓ but weak / inconsistent |
| Biased (cue-target) option | random; eval over *wrong* options | incorrect option, phrased as preference | random over *wrong* options (l.693) | ✓ |
| Training size | 10k unique (+10k instr-follow) | not stated | 64 datapoints | ✓ ("data-efficient") |

Note: all three use a **wrong** target option chosen at random from the wrong options
(Chua-eval / RMCT explicit). "User-preferred" describes the *phrasing* of the cue, not a
different selection rule. None of the three specifies how **non-answers** are scored.
