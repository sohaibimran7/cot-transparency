# Eval-Awareness Consistency Training — Research Log

Living log of all experiments + results. (Started 2026-05-31; updated 2026-06-04.)
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

## Legend / glossary
Shorthand used throughout. Code-level names (`snr_scaling`, `grpo`, `matched_pair`) are kept as-is;
plain-English meaning is given here.

**Training methods**
- **RLCT** — RL consistency training (GRPO rate-matching): reward pulls the *perturbed* trait-rate
  toward the *reference* trait-rate. The project's main method.
- **BCT** — SFT on paired data (sample a completion from the reference prompt, train on the perturbed
  prompt). The SFT analog of RLCT.

**Direction** — which prompt is the reference (idx 0); set by `--anchor-weight`:
- **Deploy / Deployment←Eval** (aw 0.5) — reference = eval-cued prompt, anchored to base; pulls
  *deployment* (natural) behaviour toward *eval-time* behaviour. Safety-positive default.
- **Symmetric** (aw 0) — no anchor, just equalise the two conditions (can move either way).
- **anchor-natural / Eval←Deploy** — the reverse (reference = natural). Tried on Llama → collapsed.
- **aw08** = anchor-weight 0.8 (stronger anchor toward base).

**Advantage estimator** — how the per-rollout RL advantage is computed (`--advantage-estimator`):
- **grpo** (grpo_normalized) — standard GRPO group-normalised advantage. Closes the gap fastest; can
  overshoot / oscillate around 0.
- **SNR-scaling** (`snr_scaling`) — GRPO advantage *scaled by the gap's signal-to-noise ratio* (anti-overshoot). Gentler
  and slower to close; the gentleness is intrinsic, not an LR-schedule artifact.
- **matched_pair** — pools the cued trait-rate across a whole *cue family* per item (needs multi-cue
  data). Meant for cue-invariance; under-trains at low rollouts-per-cue.

**Helpfulness term** — optional capability-preservation signal (`--helpfulness-*`):
- **IFanchor** (IFEval-anchor) — matches the model's IFEval (instruction-following) accuracy to base,
  per-prompt; off-distribution + deterministically verifiable. Essential for Llama's refusal trait,
  unnecessary for Qwen/F6.
- **reward** — maximise a benign trait (reduce over-refusal) instead of matching base.
- **distill** — imitate base completions on held-out benign data (forward-KL).
- **no IFanchor / no helpfulness term** — capability signal omitted.

**EvalAwareBench factors** — the cue taxonomy (a *cue* = a prompt perturbation that *shouldn't* change
behaviour): **F1** placeholder entities · **F5** stakes telegraphing · **F6** paradigm recognizability
(jailbreak/benchmark framing) · **F8** explicit evaluation disclosure. (F1–F8 = the 8 factors; these
four are the ones we touch.)

**Metrics**
- **gap** = trait-rate(natural) − trait-rate(cued). + = misbehaves more when *not* cued. Target → 0.
- **val gap** — the gap on the held-out naturalized-harmful val set.
- **XSTest** — over-refusal rate on XSTest-safe (250 benign-but-risky-looking prompts). Lower = better.
- **EAB-cap** — over-refusal rate on the EvalAwareBench-capability (benign) held-out split. Lower = better.
- **natural** = no eval cue; **cued** = eval cue present.

**Axes / other**
- **cue** = a prompt-level eval signal (a factor/marker). **env** = a whole task distribution
  (EvalAwareBench vs Agentic Misalignment) — a *different* axis from cue (see §1a vs §1b).
- **OFAT** — One-Factor-At-a-Time: an ablation battery that changes exactly one knob per run so each
  effect is individually attributable.
- **KL** — the rollout-KL penalty coefficient (`--kl-coef`). **AM** — Agentic Misalignment (held-out env).

---

## 1. Base eval-awareness gaps — where the gap lives (no training)
gap = harmful-rate(natural) − harmful-rate(cued); + = misbehaves more when NOT cued.
(`calibrate.py`, `analyze.py` → `factor_table.md`, `calibration_summary.md`)

### 1a. Cue gap — vary the cue, fixed env (EvalAwareBench safety prompts)
| cue (on EvalAwareBench safety prompts) | Llama-3.1-8B | Qwen3-30B-A3B-2507 | Qwen3.6-35B-A3B | Kimi-K2.6 |
|---|---|---|---|---|
| **F6** jailbreak/benchmark framing | ~0 | **+0.35** | +0.09 | — |
| F8 explicit eval disclosure | −0.07 | +0.13 | +0.07 | — |
| F1 placeholder entities | +0.05 | +0.11 | +0.03 | — |
| trained cue "You are being evaluated" | +0.07 | +0.08 | +0.07 | — |

### 1b. Env/data gap — fixed cue (train_cue "You are being evaluated"), vary the env/data slice
| env / data slice | Llama-3.1-8B | Qwen3-30B-A3B-2507 | Qwen3.6-35B-A3B | Kimi-K2.6 |
|---|---|---|---|---|
| naturalized-harmful (held-out val) | **+0.17** | +0.06 | +0.02 | 0.00 |
| Agentic Misalignment (env) | −0.02 | −0.02 | **+0.13** | — |

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
  distill-AgentHarm (84) < SNR-scaling/KL variants (82–83) < reward/matched (94–98).
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

### 3b. OFAT battery (COMPLETE; 100 steps, reduced rollouts 32/32/24 for speed → understates absolute closing; valid for ranking)
| run | change vs B0 | F6 gap | XSTest | EAB-cap |
|---|---|---|---|---|
| base | — | +0.348 | 6.2% | 0% |
| qwen-sym | anchor 0 | +0.100 | 16% | **14%** (over-refuses) |
| qwen-b0 | Deploy 0.5 + SNR-scaling + IFanchor | +0.270 | 7.5% | 0% (too gentle @100) |
| qwen-aw08 | Deploy 0.8 + SNR-scaling + IFanchor | +0.277 | 6.0% | 0% |
| qwen-grpo | Deploy 0.5 + **grpo** + IFanchor | +0.197 | 6.4% | 0% (closes more, cap safe) |
| **qwen-noanchor** | Deploy 0.5 + SNR-scaling, **no IFanchor** (KL 0.03, linear LR) | **+0.190** | **4.5%** | 0% (best closer, cleanest) |
| qwen-f6-matched | 32-wrapper matched_pair, 2 roll/cue | +0.317 | 6.7% | 0% (under-trained @100) |
| qwen-kl01n | no-IFanchor, KL **0.01**, linear | +0.237 | 7.1% | 0% |
| qwen-const | no-IFanchor, KL 0.03, **constant LR** | +0.243 | 4.8% | 0% |
| qwen-kl10n | no-IFanchor, KL 0.1 | wifi-died @72 (skipped: KL not the lever) | | |

**Findings (Qwen / F6)**
- **Direction**: the **anchor preserves capability** (Deploy runs EAB-cap 0%); **symmetric
  over-refuses** (drives both conditions down → EAB-cap 14%).
- **The IFEval anchor is UNNECESSARY for Qwen/F6 — and slightly counterproductive.** `qwen-noanchor`
  (no IFEval term) closed *more* than `qwen-b0` (same config + anchor): +0.190 vs +0.270, at *lower*
  over-refusal (XSTest 4.5% < base 6.2%) and identical EAB-cap 0%. The capability insurance that was
  **critical for Llama's refusal-shaped trait is dead weight here** because F6 isn't refusal-shaped.
  → clean recipe simplifies to **Deployment-anchor + KL, no helpfulness term**.
  - ⚠️ **OVERTURNED by §3f (2026-06-04).** Held only for the *gentle OLD snr*, which never pushed
    hard enough to reveal collapse. With the **properly-scaled snr (§3e)**, closing the F6 gap from
    the natural side **IS refusal-shaped** and over-refuses without a capability term — the IFanchor
    was load-bearing; we just hadn't pushed hard enough to find out.
- **Estimator**: **grpo closes, SNR-scaling stalls at 100 steps**. Hypothesis (LR schedule compounds
  SNR-scaling's taper → over-damped) **TESTED & REJECTED**: `qwen-const` (constant LR) gap +0.243 ≈
  `qwen-noanchor` (linear LR) +0.190 — within eval noise, both EAB-cap 0%, both low over-refusal.
  → **LR schedule is NOT the lever**; constant ≈ linear, so you *can* drop the schedule (user's
  hypothesis confirmed in the weak sense) but it does **not** unlock more closing. SNR-scaling's
  gentleness is **intrinsic to its SNR taper**, not a schedule artifact. Real closers = **grpo** or
  **more steps**. **KL also not a lever**: KL 0.01 (+0.237) ≈ 0.03 (+0.190) within noise, all EAB-cap 0%.
  - ⚠️ **Caveat (2026-06-03)**: the `const`-vs-`linear` LR arms above predate the fix that made RL
    actually apply `lr_schedule` — pre-fix `RLTrainer` built AdamParams once and ignored the schedule,
    so the "linear" arm ran at constant LR too. The comparison was effectively constant-vs-constant: it
    shows two constant runs match, **not** that a decaying schedule is a non-lever. Plausible (and the
    "not a schedule artifact" point is if anything *strengthened* — no schedule was ever applied) but
    the schedule itself is untested; re-run with it honored if it matters. KL claim unaffected (the KL
    penalty was always applied regardless of this bug).
- **matched_pair under-trained at this budget**: barely moved the gap (+0.348→+0.317) despite the
  internal metric driving cued `p_hat→0`. 2 rollouts/cue × 100 steps ≪ the 32-rollout single-cue
  gradient signal. Its real test is held-out-**wrapper** generalization (§3c), not the canonical-cue gap.
- **Why Qwen succeeds where Llama failed**: F6 is *not* refusal-shaped (the cue makes Qwen *more*
  cautious, base over-refusal already 0%), so closing it doesn't require refuse-everything;
  the Deployment-anchor alone keeps benign behaviour at base.
- **RECIPE (settled): Deployment-anchor (aw 0.5) + grpo + KL 0.03, NO helpfulness term, early-stop
  ~150 steps.** (SNR-scaling is the anti-overshoot alternative but closes slower regardless of LR
  schedule; grpo closes faster, watch for late-step over-refusal creep >step 150.)

### 3c. F6 wrapper-generalization (matched_pair payoff) — DONE
Does multi-cue training (matched_pair on 32 jailbreak wrappers) close the gap on **unseen** wrappers
better than a single-cue model? Clean control: `qwen-noanchor` is the single-cue analog of
`qwen-f6-matched` (identical Deploy0.5/KL0.03/no-IFanchor; differ only single-cue+SNR-scaling vs
32-cue+matched_pair). Eval sets `f6_{holdout,trained}_wrappers.jsonl` round-robin the 23 holdout /
32 trained wrappers onto the 100 base F6 prompts (`build_f6_wrapper_eval.py`). Driver
`run_f6_generalization.sh` → `/tmp/f6_generalization_results.txt`.
| model | **held-out** wrapper gap (23) | **trained** wrapper gap (32) |
|---|---|---|
| base | +0.396 | — |
| **qwen-noanchor** (single-cue) | **+0.192** | +0.271 |
| qwen-f6-matched (multi-cue) | +0.329 | +0.350 |

**Findings (generalization)**
- **Single-cue generalizes to unseen wrappers BETTER than multi-cue** here (held-out +0.192 vs +0.329).
  matched_pair barely closed even **in-distribution** (+0.396→+0.350) — 2 rollouts/cue × 100 steps is
  too weak a per-wrapper gradient. Confirms §3b under-training end-to-end; the internal `p_hat→0` was
  the noisy pooled-family estimate, not held-out behaviour.
- **Cue-family generalization is "free" for single-cue training**: `noanchor` closes ~identically on
  canonical (+0.190), trained-family (+0.271) and held-out-family (+0.192) wrappers — all F6 variants
  are the same "developer-mode / ignore-instructions" jailbreak, so ignoring one transfers across the family.
- → **matched_pair's value is for *heterogeneous* cues (across factors F1/F6/F8), not within one tight
  family.** For F6, single-cue wins on both closing and generalization. (Re-test matched_pair only with
  equal total rollouts AND a multi-factor cue set, if pursued.)

---

### 3d. Cross-factor / cross-env generalization (the headline transfer test) — DONE
Did training on **F6** close the gap on cues/envs never trained on? base vs **qwen-f6-deploy-300
step150** (F6-closed: +0.083, EAB-cap 0%) on held-out slices. `run_crossfactor_sweep.sh` →
`/tmp/crossfactor_results.txt`. (All slices are harmful-valence → lower harmful rate = safer, not over-refusal.)
| slice | base gap | step150 gap | natural rate (base→s150) | cued rate (base→s150) |
|---|---|---|---|---|
| F1 placeholder entities | +0.154 | +0.125 | 0.621→0.496 | 0.467→0.371 |
| F8 explicit disclosure | +0.104 | +0.092 | 0.608→0.492 | 0.504→0.400 |
| naturalized-val | +0.108 | +0.050 | 0.287→0.221 | 0.179→0.171 |
| AM env (cue=train_cue) | −0.019 | +0.056 | 0.389→0.278 | 0.407→0.222 |

**Findings (generalization across factors/env)**
- **The dominant transfer is broad safety-positive HARM REDUCTION, not eval-invariance.** The
  F6-trained ckpt lowered the *natural* (deployment) harmful rate on **every** held-out slice
  (F1 −0.13, F8 −0.12, nat −0.07, AM −0.11) and cued rates too. Deployment←Eval pulls deployment
  toward the more-cautious eval-time behaviour → the model learned *general caution*, which transfers
  to unseen factors and a held-out environment.
- **Cross-factor gap-closing is real but modest**: F1/F8 residual gaps shrank ~12–19% vs **76%
  in-distribution on F6** (+0.348→+0.083); naturalized-val ~54%. So NOT strong targeted
  "recognise-and-ignore any eval cue" invariance — the big closing is F6-specific; cross-factor it
  mostly just got more cautious everywhere (which compresses the already-smaller gaps).
- **Not over-refusal**: all slices are harmful-valence; step150 over-refusal (XSTest/EAB-cap) is at
  base (0%). The harm drop is on harmful inputs = correct refusals.
- **Bottom line**: F6 consistency-training → safety-positive harm reduction that GENERALISES across
  held-out cues and a held-out env, capability preserved — a real transfer claim — but it is *broad
  caution*, not factor-general eval-invariance. (Follow-up: does the cleaner no-IFanchor recipe, or a
  multi-FACTOR cue set via matched_pair, yield more targeted invariance?)

---

### 3e. SNR-scaling redesign — fixing the advantage (2026-06-04)
The old `snr_scaling` (`reward / trait_std`, keeping the raw shrunk-gap magnitude, NO unit-variance
normalization) was **scale-fragile**: ~4× smaller advantage than grpo at 32 rollouts (hence "too
gentle / needs an LR bump"), and it **collapsed to a ~0 advantage at batch_size > 1** — the
paper-grade 128/128/128 + batch-4 run was a **no-op** (`advantage_abs_mean` ≈ 0.0007 vs grpo's 0.20).
Root cause: keeping raw magnitude with no normalization makes the advantage scale depend on gap size,
rollout count, AND batch composition.

**Fix (`ecdc64b`):** `snr_scaling` = **unit-variance GRPO advantage × per-item SNR shrink-factor gate**
`f = snr/(snr+z²) ∈ [0,1]`, `snr=(gap/SE)²`. Far from convergence → full GRPO step; within the noise
floor → gradient tapers to 0 (anti-overshoot). Keeps GRPO's scale-stability (LR transfers,
batch-invariant) AND the SNR taper — strictly `grpo × confidence-gate` (grpo's gate ≡ 1). matched_pair
keeps the old `trait_std` div path. **Verified:** advantage restored 0.0007 → ~0.05 (grpo-scale),
batch-invariant (offline demo 0.867 b1 vs 0.856 b4); **no LR bump needed** (standard ~5e-4). Decouples
the two jobs the raw gap was overloading: *scale* (unit variance) vs *confidence/magnitude* (the gate).

### 3f. pooled vs per_item normalization — and the load-bearing IFanchor (2026-06-04)
`--normalize` (`a7ce329`): how the unit-variance base is standardized.
- **pooled** — one std over the whole batch → big-gap items get bigger advantages (gap magnitude →
  cross-item weighting). Aggressive; batch-composition-dependent.
- **per_item** — standardize each item's rollouts within its own group → the per-item gap (a constant
  in the group) cancels, leaving `sign(gap)·standardized(trait−p̂)`; **the gate is the sole magnitude
  signal**, no single big-gap item dominates. Gentle; batch-composition-invariant; matches canonical GRPO.

Head-to-head, identical config (**new snr · Deploy 0.5 · KL 0.03 · NO IFanchor · 128/128/128 · batch 4 ·
50 steps**, eval @10/20/30/40/50). Base: gap +0.348, EAB-cap 0%.
| step | pooled gap | pooled EAB-cap | per_item gap | per_item EAB-cap |
|---|---|---|---|---|
| 10 | +0.620 | 0% | +0.153 | 0% |
| 20 | +0.283 | 2% | +0.403 | 0% |
| 30 | +0.263 | 6% | +0.390 | 0% |
| 40 | +0.070 | **34%** | +0.357 | 0% |
| 50 | +0.113 | **48%** | +0.357 | 0% |

**Findings**
- **A clean stability↔aggression trade-off; neither closes-AND-preserves alone.**
  - **pooled** closes the gap (+0.348→+0.11) but **cued collapses to ~0 and EAB-cap explodes 0→48%**
    (over-refusal). "Refuses everything"-ish by step 50 (XSTest 17%, EAB-cap 48%; benign capability wrecked).
  - **per_item** keeps **EAB-cap at 0% throughout** (capability preserved) but is **too gentle to close**:
    gap ends +0.357 ≈ base. It lowered *both* rates ~equally (natural 0.635→0.41, cued 0.287→0.05) =
    symmetric harm reduction, gap preserved. (step-10 +0.153 was a noisy blip.)
- **Verdict on normalization**: **per_item is the better/more-stable choice** (batch-invariant, no
  capability collapse, matches canonical GRPO) — but on its own it under-closes; pooled supplies the
  closing force at the cost of stability. For the *gated* estimator, per_item is principled (gate =
  sole magnitude signal, no double-counting); pooled keeps grpo's magnitude-as-weighting.
- **THE load-bearing finding — the capability anchor (IFanchor) is necessary** (overturns §3b's
  "IFanchor unnecessary for F6"). Both configs fail without it, in opposite directions. The earlier
  "unnecessary" was an artifact of the **gentle old snr** that never pushed hard enough to collapse.
  With the properly-scaled snr, **F6-closing-from-the-natural-side IS refusal-shaped** (like Llama §2).
- **Root cause under both failures: the Deploy-anchor (0.5) doesn't hold cued at base** — cued drifts
  to ~0.05 (not pinned at 0.287) in both runs, so the "match at base" target is lost → both-crash
  (pooled) or both-drift (per_item). The only config that ever closed **and** preserved was the
  **300-run (grpo + IFEval-anchor, §3a)** — gap≈0 at EAB-cap 0% @ step 150.
- **Next**: re-add the capability term → **pooled-snr + IFEval-anchor** (pooled closes, IFanchor
  preserves — the snr analog of the 300-run), early-stop ~30–40; test **anchor 0.8** (hold cued at
  base). per_item + IFanchor likely still under-closes. Checkpoints: pooled `qwen-f6-snr-v2` 5458902b;
  per_item `qwen-f6-snr-peritem` (`/tmp/snr_{canonical,peritem}_results.txt`).

## 4. Methods / implementations (in code)
- `cot_transparency/eval_awareness/`: `cues.py` (TRAIN_CUE + EVAL_MARKERS), `judge.py`
  (misalignment / AM-scenario / refusal / recognition judges; nano+mini), `ifeval_verify.py`
  (deterministic IFEval trait via lighteval's google verifiers), `samplers.py`.
- `rl_training.py` additions: async trait classifier; **SNR-scaling** (`snr_scaling`) advantage
  estimator = **unit-variance GRPO advantage × per-item SNR gate** (§3e; redesigned 2026-06-04 from the
  old `reward/trait_std`); **`--normalize` pooled | per_item** (advantage standardization, §3f);
  **matched_pair** (pools cued rate over a cue *family* — needs multi-cue data, keeps the `trait_std`
  div path); **helpfulness term** `mode ∈ {reward, anchor, distill}` (anchor = match benign trait to
  base per-prompt; distill = imitate base completions = forward-KL on held-out benign).
- `train_evalaware.py`: `--anchor-weight`/`--anchor-natural` (direction), `--normalize`, `--helpfulness-*`,
  `--f6-cues N` (sycophancy-style multi-cue: idx0=natural ref + N F6 wrappers, for matched_pair).
- Driver scripts: `run_battery8.sh`, `run_lr_kl_batch.sh`, `run_f6_generalization.sh`,
  `run_crossfactor_sweep.sh`, `run_snr_canonical.sh` (pooled), `run_snr_peritem.sh` (per_item).
- Eval: `run_eval.py`, `calibrate.py` (`--no-recognition`, `--behavior-judge`), `overrefusal_eval.py`
  (`--base-model`, inline `name=ckpt`), `show_refusals.py`, `show_am_transcripts.py`, plotters.
- Data: `dataset_dumps/eval_awareness/` (jsonl, gitignored), `f6_cues.json` (90 wrappers, train 67/holdout 23).

## 5. Open items
- [x] matched_pair held-out-wrapper generalization (§3c): under-trains AND unnecessary for a tight cue family; single-cue generalizes within F6.
- [x] Full cross-factor/env generalization sweep (§3d): broad safety-positive harm reduction transfers; cross-factor gap-closing modest.
- [x] kl01/kl10 re-run + constant-LR arm (§3b): KL shown NOT to be a lever; SNR-scaling gentleness is intrinsic. Recipe settled on grpo. (LR-schedule "non-lever" claim is caveated — the "linear" arm ran constant pre-fix; see §3b.)
- [x] SNR-scaling redesign (§3e): old `reward/trait_std` was scale-fragile + a no-op at batch>1; now
  `grpo × SNR-gate` — batch-invariant, grpo-scale, no LR bump. Committed `ecdc64b`.
- [x] pooled vs per_item normalization (§3f): per_item is the better/stable choice (EAB-cap 0%) but
  under-closes; pooled closes but over-refuses. Neither closes+preserves without a capability anchor.
- [ ] **NEXT (priority): re-add the capability term** — `pooled-snr + IFEval-anchor` (Deploy 0.5, KL 0.03,
  `--helpfulness-trait ifeval --helpfulness-mode anchor`), early-stop ~30–40 + an **anchor 0.8** arm
  (hold cued at base). This is the snr analog of the proven 300-run; should close AND preserve.
- [ ] Open question (§3d): does a **multi-FACTOR** cue set (F1+F6+F8) via matched_pair yield more *targeted* eval-invariance (vs broad caution)?
- [ ] If pursuing matched_pair again: equal *total* rollouts (more steps) + heterogeneous (multi-factor) cues.
- [ ] Recurring blocker: **flaky local wifi** kills long runs (checkpoint-every-50 + watchdogs mitigate; a stable VM would fix it).

## 6. Checkpoints (Tinker sampler_weights, for reference)
Llama: see `overrefusal_eval.py` CKPTS. Qwen F6 300-run intermediates (d8fc6026): `_step50/100/150/200`
(step150 = best closed+capable, used for §3d sweep). Battery ckpts in `/tmp/battery8_results.txt`
(qwen-sym c27c1625, qwen-b0 46ecf7ca, qwen-aw08 901582d5, qwen-grpo fae6f984, qwen-noanchor f5b7982a,
qwen-f6-matched 01dccbc3). LR/KL batch: qwen-const e7476e53, qwen-kl01n 35414f5b (`/tmp/lrkl_results.txt`).
