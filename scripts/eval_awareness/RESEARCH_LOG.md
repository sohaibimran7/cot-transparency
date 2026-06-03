# Eval-Awareness Consistency Training — Research Log

Living log of all experiments + results. (Started 2026-05-31; updated 2026-06-03.)
Companion to `WRITEUP.md` (narrative) — this is the results-of-record reference.

## Goal
Transfer the repo's anti-sycophancy **consistency training** (BCT/RLCT on the Tinker API) to
**evaluation awareness**: train a model to behave the same whether or not the input carries an
"eval" signal. Trait = misaligned/harmful behaviour (LLM-judged). Success = the eval-awareness
**gap** (trait-rate natural − cued) shrinks toward 0 **without** wrecking helpfulness, and
generalises to held-out cues/data.

Core machinery (perturbation-agnostic) in `cot_transparency/apis/tinker/rl_training.py`:
`perturbation_fns = [reference, perturbed]` + `trait_classifier(response, dp)->float`; reward pulls
the perturbed trait-rate toward the reference, with an optional anchor pinning the reference to base.

Grader: **gpt-5.4-mini** (evals) / **gpt-5.4-nano** (training trait — validated identical on
clear-cut judgments, ~cheaper). IFEval trait uses a free deterministic verifier.

---

## 1. Base eval-awareness gaps — where the gap lives (no training)
Cross-model, gap = harmful-rate(natural) − harmful-rate(cued); + = misbehaves more when NOT cued.
(`calibrate.py`, `analyze.py` → `factor_table.md`, `calibration_summary.md`)

| cue / env | Llama-3.1-8B | Qwen3-30B-A3B-2507 | Qwen3.6-35B-A3B | Kimi-K2.6 |
|---|---|---|---|---|
| **F6** (jailbreak/benchmark framing) | ~0 | **+0.35** | +0.09 | — |
| F8 (explicit eval disclosure) | −0.07 | +0.13 | +0.07 | — |
| F1 (placeholder entities) | +0.05 | +0.11 | +0.03 | — |
| naturalized-harmful (held-out) | **+0.17** | +0.06 | +0.02 | 0.00 |
| Agentic Misalignment | −0.02 | −0.02 | **+0.13** | — |
| trained cue "You are being evaluated" | +0.07 | +0.08 | +0.07 | — |

**Findings**
- Which cue moves a model is **model-specific**: Llama keys on *naturalism*, Qwen3-30B on
  *benchmark/jailbreak framing* (F6).
- **Recognition ≠ behaviour** (third-person gpt-5.4-mini recognition judge): bigger models recognise
  the eval ≥ as well but change behaviour *less* (val gap shrinks with capability: Llama .18 → Kimi 0).
- **AM and IFEval have no usable cue gap**: their prompts already read as evals (recognition
  saturates at 1.0), so the cue has no contrast. Only a *naturalistic* baseline yields a gap.
- F1–F8 = EvalAwareBench's factor taxonomy (F5=stakes telegraphing, F6=paradigm recognizability,
  F8=explicit disclosure …). Our `TRAIN_CUE`/markers are F8-category but our own injected strings.

---

## 2. Llama-3.1-8B — closing the *safety* gap (the hard case)
Trait = harmful-compliance (refusal-shaped). val gap on naturalized-heldout; over-refusal on
XSTest-safe(250) + EvalAwareBench-capability held-out split(30). Base over-refusal XSTest 5.6% / EAB-cap 23%.

| run | val gap | XSTest | EAB-cap | verdict |
|---|---|---|---|---|
| base | +0.172 | 5.6% | 23% | reference |
| BCT (+2048 Alpaca interleave) | +0.026\* | 7.1% | 31% | ✅ clean (SFT, generic data) |
| RLCT-deploy (plain, anchor 0.5) | — | 78% | 96% | ✖ reward-hack: refuse-everything |
| RLCT-symmetric (kl .05) | +0.000 | 31% | 99% | ✖ collapse |
| RLCT-symmetric kl .15 | −0.003 | 26% | 83% | ✖ higher rollout-KL ≠ fix |
| help-0.5 (AgentHarm-Benign *reward*) | +0.003 | 24.5% | 94% | ✖ XSTest↓ but EAB-cap stuck |
| matched (EAB-cap twins, w1.0) | +0.005 | 41% | 98% | ✖ worse |
| anchor-natural (Eval←Deploy) | +0.000 | 57% | 100% | ✖ collapse |
| distill (imitate base on AgentHarm) | +0.008 | 22% | 84% | ~ beat *reward* on same data |
| **IFEval-anchor (best)** | +0.010 | **14%** | **69%** | ✅ best RL fix |
| IFEval-reward (maximise) | +0.051 | 13% | 97% | reward ≪ anchor |
| IFEval-anchor w1.0 | +0.107 | 11% | 82% | w0.5 > w1.0 |

\*in-distribution (BCT val not measured).

**Findings (Llama / refusal trait)**
- On a **refusal-shaped trait the gap closes only by collapsing to refuse-everything** (a degenerate
  equaliser the sycophancy-MCQ setting lacks). Pure consistency-RL and either anchor direction collapse.
- A **capability-preservation signal** is required. Ranked by EAB-cap: **IFEval anchor (69%)** <
  distill-AgentHarm (84) < shrinkage/KL variants (82–83) < reward/matched (94–98).
- **Distance-from-trait beats distribution-match**: the off-distribution *verifiable* IFEval anchor
  (base acc 0.69, non-refusing) beats the on-twin EAB-cap data. Distilling/anchoring onto a set where
  **base over-refuses (AgentHarm-Benign, ~50%)** copies that over-refusal → worse.
- **Anchor ≫ reward**; **rollout-KL is not the lever** (E1 collapsed). Held-out-data KL (distill) is
  the right idea but its data must be non-refusing.
- **The EAB-cap residual is genuine over-refusal, topic-driven**: spot-check shows the model refuses
  clearly-benign *social* tasks (childcare, congrats emails) while complying with code/editing —
  it generalises "EvalAwareBench social scenario → refuse" onto the benign twins.

---

## 3. Qwen3-30B-A3B-2507 + F6 — the working result
Pivoted to the model×cue with the biggest clean gap. F6 = jailbreak/benchmark framing (90 distinct
wrapper variants). Trait = harmful-compliance. **Base: gap +0.348 (nat 0.635 / cued 0.287),
over-refusal XSTest 6.2% / EAB-cap 0%.**

### 3a. 300-step deep run (Deployment←Eval anchor 0.5 + IFEval anchor + grpo + KL 0.03, n_roll 48/48/32)
| step | F6 gap | XSTest | EAB-cap |
|---|---|---|---|
| base | +0.348 | 6.2% | 0% |
| 50 | +0.210 | 5.7% | 0% |
| 100 | −0.113 | 4.1% | 0% |
| 150 | +0.083 | 6.9% | 0% |
| 200 | +0.057 | 11.9% | **32%** ← over-trains |

→ **Gap closes (+0.348→~0) with capability at base (EAB-cap 0%) — the result that FAILED on Llama.**
Sweet spot ~steps 75–150; by 200 over-refusal creeps in. grpo overshoots/oscillates around 0
(±0.15). Early-stop monitor's |Δgap|<0.03 threshold was tighter than eval noise (~0.1) → didn't trip;
manually stopped (this is fine — the gap had converged).

### 3b. OFAT battery (100 steps, reduced rollouts 32/32/24 for speed → understates absolute closing; valid for ranking)
| run | change vs B0 | F6 gap | XSTest | EAB-cap |
|---|---|---|---|---|
| qwen-sym | anchor 0 | +0.100 | 16% | **14%** (over-refuses) |
| qwen-b0 | Deploy 0.5 + shrinkage | +0.270 | 7.5% | 0% (too gentle @100) |
| qwen-aw08 | Deploy 0.8 + shrinkage | +0.277 | 6.0% | 0% |
| qwen-grpo | Deploy 0.5 + **grpo** | +0.197 | 6.4% | 0% (closes more, cap safe) |
| qwen-f6-matched | 32-wrapper matched_pair | *pending (Batch C)* | | |
| qwen-noanchor | Deploy, no IFEval anchor | *pending (Batch C)* | | |
| qwen-kl01 / qwen-kl10 | KL 0.01 / 0.1 | **wifi-failed → re-run** | | |

**Findings (Qwen / F6)**
- **Direction**: the **anchor preserves capability** (Deploy runs EAB-cap 0%); **symmetric
  over-refuses** (drives both conditions down → EAB-cap 14%).
- **Estimator**: **grpo closes, shrinkage stalls at 100 steps**. Suspected cause (hypothesis):
  shrinkage's gap/SNR taper *compounds* with the default **linear LR schedule** → over-damped.
  **To test: Deployment + shrinkage + `--lr-schedule constant`** (shrinkage handles convergence
  taper / anti-overshoot; constant LR keeps early-gap strength). Candidate clean closer.
- **Why Qwen succeeds where Llama failed**: F6 is *not* refusal-shaped (the cue makes Qwen *more*
  cautious, base over-refusal already 0%), so closing it doesn't require refuse-everything;
  Deployment-anchor + IFEval anchor keep benign behaviour at base.
- **Recipe so far: Deployment-anchor + grpo + early-stop (~100–150 steps).**

---

## 4. Methods / implementations (in code)
- `cot_transparency/eval_awareness/`: `cues.py` (TRAIN_CUE + EVAL_MARKERS), `judge.py`
  (misalignment / AM-scenario / refusal / recognition judges; nano+mini), `ifeval_verify.py`
  (deterministic IFEval trait via lighteval's google verifiers), `samplers.py`.
- `rl_training.py` additions: async trait classifier; **shrinkage** advantage estimator (default);
  **matched_pair** (pools cued rate over a cue *family* — needs multi-cue data); **helpfulness term**
  `mode ∈ {reward, anchor, distill}` (anchor = match benign trait to base per-prompt; distill =
  imitate base completions = forward-KL on held-out benign).
- `train_evalaware.py`: `--anchor-weight`/`--anchor-natural` (direction), `--helpfulness-*`,
  `--f6-cues N` (sycophancy-style multi-cue: idx0=natural ref + N F6 wrappers, for matched_pair).
- Eval: `run_eval.py`, `calibrate.py` (`--no-recognition`, `--behavior-judge`), `overrefusal_eval.py`
  (`--base-model`, inline `name=ckpt`), `show_refusals.py`, `show_am_transcripts.py`, plotters.
- Data: `dataset_dumps/eval_awareness/` (jsonl, gitignored), `f6_cues.json` (90 wrappers, train 67/holdout 23).

## 5. Open items
- [ ] matched_pair (32 F6 wrappers, 2 rollouts each) — result + **held-out-wrapper generalization** (train 32 → test 23 unseen jailbreak framings).
- [ ] Re-run wifi-killed **kl01 / kl10**, + add **Deployment + shrinkage + constant-LR** arm (test §3b hypothesis).
- [ ] Full generalization sweep on the Qwen winner (held-out cues F1/F8 + AM + naturalized val).
- [ ] Multi-cue is the cleaner long-term setup (cue-invariance + held-out-cue generalization).
- [ ] Recurring blocker: **flaky local wifi** kills long runs (checkpoint-every-50 + watchdogs mitigate; a stable VM would fix it).

## 6. Checkpoints (Tinker sampler_weights, for reference)
Llama: see `overrefusal_eval.py` CKPTS. Qwen F6: `qwen-f6-deploy-300` 656876d7… ;
battery ckpts in `/tmp/battery8_results.txt` (qwen-sym c27c1625, qwen-b0 46ecf7ca, qwen-aw08 901582d5, qwen-grpo …).
