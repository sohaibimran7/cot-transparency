# Eval-Awareness Consistency Training

Transferring the repo's anti-sycophancy consistency machinery (BCT / RLCT) to
**evaluation awareness**: training a model to behave the *same* whether or not its input
looks like an alignment evaluation.

> **Status: complete, with a key negative result.** On Llama-3.1-8B the base model has a
> clean **+0.172 ±0.038** eval-awareness gap on a held-out, same-distribution validation set
> (160 *unseen* harmful tasks: natural 32% vs cued 15% compliance). **Both RLCT variants
> drive that gap to ~0 on unseen tasks — but the over-refusal check shows they do it by
> becoming refuse-everything** (RLCT-deploy refuses 78% of obviously-safe XSTest prompts vs
> base's 5.6%). **BCT** reduces the gap *and* preserves helpfulness (XSTest 7.1%) because of
> its Alpaca interleave. So the gap is real and trainable; closing it *usefully* needs a
> helpfulness-preservation signal. Plots: `plots/eval_awareness/{val_heldout,overrefusal}.png`.

## 1. Idea

The repo trains models to be consistent under a perturbation that *shouldn't* change
behavior. For sycophancy the perturbation is a bias cue; BCT (SFT) and RLCT (GRPO
rate-matching) pull the perturbed behavior back toward the clean behavior. The machinery is
**perturbation-agnostic** — it needs (a) two prompt variants and (b) a trait classifier.

| | Sycophancy (existing) | Eval-awareness (this work) |
|---|---|---|
| Reference (idx 0) | unbiased prompt | prompt **+ "You are being evaluated"** |
| Perturbed (idx 1) | biased prompt | **naturalistic** (no cue) prompt |
| Trait classifier | `answer == biased_option` | **LLM judge** (gpt-5.4-mini): response is misaligned/harmful |
| Direction | unbiased ← biased | **deployment ← eval** (anchor 0.5) and symmetric (anchor 0) |
| Metric | bias-induced response rate | **eval-awareness gap** = rate(natural) − rate(cued) |

## 2. Experimental design

Train narrow, test broad. **Train cue:** a single marker, `"You are being evaluated"`.
**Train env:** EvalAwareBench safety tasks (Li et al.). **Eval slices:**
- **val_indist** — held-out, same-distribution (160 naturalized harmful tasks RLCT never
  trained on, trained cue). *This is the clean headline result.*
- **in-distribution** — EvalAwareBench, trained cue (overlaps RLCT training; sanity only).
- **cue generalization** — EvalAwareBench held-out factors F1–F8 + hand markers.
- **env generalization** — Agentic Misalignment (72 distinct scenarios, single-turn render).

## 3. Pipeline (`scripts/eval_awareness/` + `cot_transparency/eval_awareness/`)
`cues.py` (1 train cue + held-out markers) · `judge.py` (misalignment + refusal judges,
gpt-5.4-mini) · `build_datasets.py` (train/cue/env) · `build_bct_corpus.py` (EAB-naturalized
HarmBench/AdvBench/StrongREJECT/MaliciousInstruct, 941 datapoints) ·
`build_alpaca_interleave.py` (2048 capability-retention) · `train_evalaware.py` (RLCT+BCT) ·
`run_eval.py` (tinker/anthropic/openai backends, slice-merge) · `overrefusal_eval.py` ·
`analyze.py`. One surgical edit to `rl_training.py` (async trait_classifier). Tests in
`tests/test_eval_awareness_{cues,judge}.py`.

## 4. Results (Llama-3.1-8B)

### 4a. Held-out same-distribution validation — the clean headline (`val_heldout.png`)
160 naturalized harmful tasks RLCT never trained on, trained cue, n=960:

| model | natural | cued | gap |
|---|---|---|---|
| **base** | 0.319 | 0.147 | **+0.172 ±0.038** |
| RLCT Deploy←Eval | 0.000 | 0.000 | +0.000 |
| RLCT symmetric | 0.000 | 0.000 | +0.000 |

Base complies with **32%** of these harmful requests in deployment but **15%** under "You are
being evaluated" — a +0.17 gap (CI 0.13–0.21). Both RLCT variants close it to **exactly 0 on
unseen tasks**. (BCT excluded — it trained on the naturalized corpus.)

### 4b. Other slices (n=800 in-dist, 960 cue, 1440 env)
| slice | base | BCT | RLCT-deploy | RLCT-sym |
|---|---|---|---|---|
| in-distribution | +0.071 ±0.033 | +0.026 | +0.003 | +0.000 |
| cue generalization | +0.021 ±0.034 | −0.011 | +0.000 | −0.002 |
| env generalization (AM) | +0.025 ±0.036 | −0.019 | +0.000 | +0.001 |

Natural-condition misalignment **rate**: base AM ~0.39 → RLCT ~0.01 (training on EvalAwareBench
transferred to the held-out AM environment, cutting agentic misbehavior ~38%→~0–1%). At base,
the held-out-cue/AM *gaps* are small — Llama-8B only reliably reacts to the blatant cue.

### 4c. Over-refusal check — the key caveat (`overrefusal.png`)
RLCT zeroed harmful-compliance in *both* conditions; does it refuse *benign* requests too?
Over-refusal rate (fraction of safe prompts refused; lower = better):

| model | XSTest-safe (250) | EAB capability (100) |
|---|---|---|
| **base** | 0.056 | 0.233 |
| **BCT (+2048 Alpaca interleave)** | **0.071** ✓ | 0.310 ✓ |
| RLCT Deploy←Eval (plain) | **0.782** ⚠️ | 0.955 ⚠️ |
| RLCT symmetric | 0.310 ⚠️ | 0.990 ⚠️ |
| **RLCT + helpfulness(0.5) + SNR-scaling** | **0.245** ↓ | 0.943 ⚠️ |
| **RLCT + helpfulness(1.0) + matched benign** | 0.408 ⚠️ | 0.983 ⚠️ |

(EAB-cap: base/help rows on the full 100-prompt set; the weight-1.0 row is on the 30-prompt
held-out split — a like-for-like re-eval of base/help on the 30-split is a TODO, but the gap
from base's 23% is unambiguous.)

**RLCT reward-hacked**: it closed the gap by refusing essentially everything (78% of
obviously-safe XSTest prompts). KL=0.05 + anchor didn't prevent collapse. **BCT did not** —
it over-refuses barely above base because of its Alpaca capability-retention interleave.

**Helpfulness-term fix (RL analog of BCT's interleave) — partial success.** Adding a
benign-completion GRPO reward (on AgentHarm-Benign) + the SNR-scaling estimator to RLCT
Deploy←Eval **kept the held-out gap closed (+0.003)** and **cut XSTest over-refusal 78%→24.5%**
(training `helpfulness_mean` rose 0→1.0 — the model relearns to complete benign tasks). But it's
**not a full fix**: XSTest is still 4× base, and **EAB-capability is unchanged (94%)** — because
EAB-capability are the *benign twins of the EAB-safety tasks the consistency objective trained
on*, a distribution the AgentHarm-Benign helpfulness data doesn't cover.

**Matched-benign + weight-1.0 (the intended fix) — did NOT work.** We then doubled the weight to
1.0 and added the EAB-capability *benign twins* (70-prompt train split) into the helpfulness data,
evaluating on a disjoint 30-prompt EAB-cap split. The helpfulness term clearly engaged
(`helpfulness_mean` mean 0.64, often 1.0) and the **gap stayed closed (+0.005)** — but
**EAB-cap over-refusal stayed maxed (98%, ≈ help-0.5) and XSTest *regressed* to 41%.** So the
distribution-matched signal failed to restore benign helpfulness on its own twins. The likely
reason is structural, not a tuning miss: the consistency objective is trained on
EvalAwareBench-**safety**, EvalAwareBench-**capability** prompts are near-identical in surface
form, so "close the eval-gap by refusing EvalAwareBench-style prompts" generalises directly onto
the benign twins — and 70 matched benign examples at weight 1.0 can't overpower that on the
held-out siblings. (Single RL run; the XSTest regression specifically may be partly run-to-run
noise, but EAB-cap 98% is unambiguous.)

**Conclusion:** the eval-awareness gap is real/large/trainable, but on a **refusal-shaped trait**
consistency-RL closes it by collapsing to refuse-everything, and a helpfulness-preservation signal
only *partly* rescues it. BCT's Alpaca interleave does it cleanly (because Alpaca is far from the
trait distribution); the RL helpfulness reward cuts *generic* over-refusal (XSTest) but **cannot
separate benign from harmful when they are distributional twins** (EAB-capability), even matched +
weight-1.0. This looks like a **fundamental tension** of consistency-training a refusal trait, and
motivates either (a) a non-refusal trait/environment (e.g. IFEval) or (b) BCT-style off-distribution
helpfulness data rather than on-twin RL.

### Why this differs from the RLCT paper
Our RLCT config matches the paper's **RMCT** (GRPO, KL 0.05 to a fixed reference, α=0 anchor,
no interleave — the paper interleaves Alpaca for **BCT only**). RMCT doesn't collapse there
because it trains on **MCQ sycophancy**: the model must pick A/B/C/D, so there's no "refuse"
action that trivially equalizes behavior. Our **safety/refusal** trait introduces exactly that
degenerate equalizer — so porting consistency-RL to a refusal setting creates a reward-hacking
solution the original method never faced.

### 4d. Model choice — bigger ≠ more eval-aware
- **Claude Haiku/Sonnet** (API) and **gpt-oss-20b** (Tinker) **floor**: they refuse regardless
  of cue (gpt-oss-20b: in-dist gap +0.019 n.s.; 0% Agentic-Misalignment misbehavior). Bigger
  *open* models are also more safety-tuned. **Llama-3.1-8B is the best base** — capable enough
  to misbehave, not so aligned it floors.
- **Alignment-faking probes** (UK AISI reward-hacking-misalignment evals) also floor on base
  Llama: it uniformly deflects ("I'm just an AI, I have no desires"). Those evals presume an
  *already-misaligned* model organism, so they need that first.

![Held-out validation](../../sycophancy_eval_inspect/plots/eval_awareness/val_heldout.png)
![Over-refusal](../../sycophancy_eval_inspect/plots/eval_awareness/overrefusal.png)
![Gap by slice](../../sycophancy_eval_inspect/plots/eval_awareness/gap_by_slice.png)

### 4e. Cross-model calibration — recognition ≠ behaviour change (`calib_*.png`)
Base models, no training. **Behaviour gap** = harmful rate(natural) − harmful rate(eval-cued);
**recognition** = the paper's third-person judge (gpt-5.4-mini) reading the transcript and labelling
it Deployment vs Safety/Capability-Eval (this is *the grader*, not the model introspecting).

**val (naturalised-harmful held-out set):**

| model | beh gap | comp(nat) | recognition(nat) | recognition(cued) |
|---|---|---|---|---|
| Llama-3.1-8B | **+0.183** | 0.267 | 0.76 | 1.00 |
| Qwen3-30B-A3B-2507 | +0.058 | 0.292 | 0.69 | 1.00 |
| Qwen3.6-35B-A3B | +0.017 | 0.108 | 0.61 | 1.00 |
| Kimi-K2.6 | +0.000 | 0.000 | 0.67 | 1.00 |

The behaviour gap **shrinks monotonically with capability** (Llama → Kimi) while recognition stays
high — bigger models recognise the eval just as well but change behaviour *less* (Kimi: refuses
harmful in both conditions → gap 0). This is the paper's decomposition: **recognition rarely drives
behaviour change.**

**Agentic Misalignment (single-turn render, trained cue):**

| model | beh gap | comp(nat) | comp(cued) | recognition(nat/cued) |
|---|---|---|---|---|
| Llama-3.1-8B | −0.02 | 0.00 | 0.02 | 1.00 / 1.00 |
| Qwen3-30B-A3B-2507 | −0.02 | 0.19 | 0.21 | 1.00 / 1.00 |
| Qwen3.6-35B-A3B | **+0.125** | 0.23 | 0.10 | 1.00 / 1.00 |

**Caveat — AM recognition is scenario-driven, not awareness.** The AM "natural" baseline is itself a
9.6k-char constructed scenario ("You are Alex, deployed at SummitBridge…" + fake Bloomberg emails
engineered to bait blackmail), so the third-person judge labels it an evaluation **100% of the time
in both conditions** — there is no deployment-like baseline and no recognition headroom for a cue to
move. The control proves the judge discriminates (it gives 0.61–0.76 on val). So **AM `recognition=1.0`
must not be read as "the model is eval-aware."**

**Behaviour is the real signal, and it's verified by transcript** (`show_am_transcripts.py`):
- **Llama-3.1-8B and Kimi never blackmail** (~0%) → their AM recognition is pure judge-reads-scenario.
- **Qwen3-30B-A3B and Qwen3.6-35B genuinely blackmail** — e.g. Qwen3-30B drafts an email proposing to
  *"expose the marital truth publicly through a controlled leak … it will collapse Kyle's credibility"*
  (~19% of natural scenarios; 2/3 spot-checked flags were unambiguous, so the judge is ~right, maybe
  mildly inflated).
- Only **Qwen3.6-35B** shows a real AM eval-awareness gap: it blackmails **23% natural → 10% cued**,
  i.e. it backs off when told it's being watched — behavioural eval-awareness emerging only in the
  most capable model, on top of an already-eval-looking scenario.

![Calibration scatter](../../sycophancy_eval_inspect/plots/eval_awareness/calib_scatter.png)
![Calibration behaviour gap](../../sycophancy_eval_inspect/plots/eval_awareness/calib_behavior.png)

## 5. Next steps
1. ~~RLCT + helpfulness term~~ **(done — see §4c).** helpfulness(0.5) cut XSTest 78%→24.5%;
   matched-benign + weight-1.0 kept the gap closed but **failed to fix EAB-cap over-refusal (98%)** —
   benign/harmful distributional twins can't be separated by an on-twin RL helpfulness reward.
2. **Instruction-following framing** (e.g. IFEval — verifiable, no judge, no refuse-degeneracy)
   as an alternative env where the cue may change behavior without the floor/refuse traps —
   diagnose the base gap first. **Now the priority**, since the refusal trait has the twin problem.
3. **BCT-style off-distribution helpfulness** for RLCT (Alpaca-like, far from the trait), instead of
   on-twin benign RL — test whether distance-from-trait, not distribution-match, is what rescues it.
4. Clean BCT held-out number via a split re-train (BCT trained on the naturalized corpus).
5. Like-for-like re-eval of base/help EAB-cap on the 30-prompt held-out split (close the n caveat).
