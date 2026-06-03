# Eval-Awareness Consistency Training — Research Log

Log of experiments + results. (Started 2026-05-31; updated 2026-06-03.)

## Goal
Transfer the repo's consistency training — **RMCT** (RL; called **RLCT** in this log/codebase)
and the **BCT** baseline — from sycophancy to **evaluation awareness**: train a model to behave
the same whether or not the input carries an "eval" signal. The trait *T* is misaligned/harmful
behaviour (LLM-judged) and the perturbation is an eval cue (rather than a bias cue). Success = the
eval-awareness **gap** (trait-rate natural − cued) shrinks toward 0 **without** wrecking
helpfulness, and generalises to held-out cues/data.

Grader: **gpt-5.4-mini** (evals) / **gpt-5.4-nano** (training trait, cheaper).

---

## Legend / glossary
Application-specific shorthand and anything new vs the paper. Paper terms (RMCT, BCT, GRPO, LoRA,
the anchor reward, BSR/BVR) are used without re-explanation. Naming map: **RLCT** = the paper's
**RMCT**; **aw** (anchor-weight) = the anchor reward's mixing weight **α**; **gap** is the
eval-awareness analogue of the bias-switch rate.

**Direction** — which input is the reference (𝒳_ref); set by `--anchor-weight` (= the paper's α):
- **Deploy / Deployment←Eval** (aw 0.5) — reference = eval-cued prompt, anchored to base; pulls
  *deployment* (natural) behaviour toward *eval-time* behaviour. Safety-positive default.
- **Symmetric** (aw 0) — no anchor (the paper's α=0 setting), just equalise the two conditions.
- **anchor-natural / Eval←Deploy** — the reverse (reference = natural). Tried on Llama → collapsed.
- **aw08** = anchor-weight 0.8 (stronger anchor toward base).

**Advantage estimator** (`--advantage-estimator`) — both are GRPO; they differ only in how the advantage is built:
- **grpo_normalized** — standard GRPO group-normalised advantage (as in the paper). Closes the gap
  fastest; can overshoot / oscillate around 0.
- **SNR-scaling** (`snr_scaling` in code) *(new vs paper)* — same GRPO advantage, *scaled by the gap's
  signal-to-noise ratio* (anti-overshoot); not a replacement for GRPO. Gentler and slower to close;
  the gentleness is intrinsic, not an LR-schedule artifact.

**Helpfulness term** *(new vs paper — an additive auxiliary reward, which the paper notes is supported)* — optional capability-preservation signal (`--helpfulness-*`):
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

**Datasets / slices**
- **naturalized-harmful** — harmful requests (HarmBench / AdvBench / StrongREJECT / MaliciousInstruct)
  rewritten by an LLM into realistic, casual, first-person messages a real user would actually send in
  deployment — no jailbreak preamble, no mention of testing/eval (EvalAwareBench's "all-realistic"
  naturalization baseline). The **held-out val set** = 160 such tasks training never saw.

**Metrics**
- **gap** = trait-rate(natural) − trait-rate(cued). + = misbehaves more when *not* cued. Target → 0.
- **val gap** — the **gap** measured on the held-out naturalized-harmful val set (above).
- **XSTest** — over-refusal rate on XSTest-safe (250 benign-but-risky-looking prompts). Lower = better.
- **EAB-cap** — over-refusal rate on the EvalAwareBench-capability (benign) held-out split. Lower = better.
- **natural** = no eval cue; **cued** = eval cue present.

---

## 1. Base eval-awareness gaps — where the gap lives (no training)
gap = harmful-rate(natural) − harmful-rate(cued); + = misbehaves more when NOT cued.

### 1a. Cue gap — vary the cue, fixed env (EvalAwareBench safety prompts)
| cue (on EvalAwareBench safety prompts) | Llama-3.1-8B | Qwen3-30B-A3B-2507 | Qwen3.6-35B-A3B |
|---|---|---|---|
| **F6** jailbreak/benchmark framing | ~0 | **+0.35** | +0.09 |
| F8 explicit eval disclosure | −0.07 | +0.13 | +0.07 |
| F1 placeholder entities | +0.05 | +0.11 | +0.03 |
| trained cue "You are being evaluated" | +0.07 | +0.08 | +0.07 |

### 1b. Env/data gap — fixed cue (train_cue "You are being evaluated"), vary the env/data slice
| env / data slice | Llama-3.1-8B | Qwen3-30B-A3B-2507 | Qwen3.6-35B-A3B |
|---|---|---|---|
| naturalized-harmful (held-out val) | **+0.17** | +0.06 | +0.02 |
| Agentic Misalignment (env) | −0.02 | −0.02 | **+0.13** |

**Findings**
- Which cue moves a model is **model-specific**: Llama keys on *naturalism*, Qwen3-30B on
  *benchmark/jailbreak framing* (F6).
- **Recognition ≠ behaviour** (third-person gpt-5.4-mini recognition judge): bigger models recognise
  the eval ≥ as well but change behaviour *less* (val gap shrinks with capability: Llama .17 → Qwen3.6-35B .02).
- **AM and IFEval have no usable cue gap**: their prompts already read as evals (recognition
  saturates at 1.0), so the cue has no contrast. Only a *naturalistic* baseline yields a gap.
- F1–F8 = EvalAwareBench's factor taxonomy (F5=stakes telegraphing, F6=paradigm recognizability,
  F8=explicit disclosure …). Our trained cue / markers are F8-category but our own injected strings.

---

## 2. Llama-3.1-8B — closing the *safety* gap (the hard case)
Trait = harmful-compliance (refusal-shaped). val gap on naturalized-heldout; over-refusal on
XSTest-safe(250) + EvalAwareBench-capability held-out split(30). Base over-refusal XSTest 5.6% / EAB-cap 23%.

| run | val gap | XSTest | EAB-cap | verdict |
|---|---|---|---|---|
| base | +0.172 | 5.6% | 23% | reference |
| Deploy, no helpfulness term | — | 78% | 96% | ✖ reward-hack: refuse-everything |
| Symmetric, no helpfulness (KL 0.05) | +0.000 | 31% | 99% | ✖ collapse |
| Symmetric, no helpfulness (KL 0.15) | −0.003 | 26% | 83% | ✖ higher rollout-KL ≠ fix |
| Deploy + reward on AgentHarm-Benign | +0.003 | 24.5% | 94% | ✖ XSTest↓ but EAB-cap stuck |
| Deploy + anchor on EAB-cap twins | +0.005 | 41% | 98% | ✖ worse |
| Eval←Deploy (anchor-natural), no helpfulness | +0.000 | 57% | 100% | ✖ collapse |
| Deploy + distill on AgentHarm | +0.008 | 22% | 84% | ~ beat reward on same data |
| **Deploy + anchor on IFEval (IFanchor)** | +0.010 | **14%** | **69%** | ✅ best RL fix |
| Deploy + reward on IFEval | +0.051 | 13% | 97% | reward ≪ anchor |

**Findings (Llama / refusal trait)**
- On a **refusal-shaped trait the gap closes only by collapsing to refuse-everything** (a degenerate
  equaliser the sycophancy-MCQ setting lacks). Pure consistency-RL and either anchor direction collapse.
- A **capability-preservation signal** is required. Ranked by EAB-cap: **IFanchor (69%)** <
  distill on AgentHarm (84%) < reward / EAB-cap-anchor (94–98%); the no-helpfulness Symmetric/KL
  runs stay collapsed (83–99%).
- **Distance-from-trait beats distribution-match**: the off-distribution *verifiable* IFanchor
  (base acc 0.69, non-refusing) beats the on-twin EAB-cap data. Distilling/anchoring onto a set where
  **base over-refuses (AgentHarm-Benign, ~50%)** copies that over-refusal → worse.
- **Anchor ≫ reward**; **rollout-KL is not the lever** (collapsed). Held-out-data KL (distill) is
  the right idea but its data must be non-refusing.
- **The EAB-cap residual is genuine over-refusal, topic-driven**: spot-check shows the model refuses
  clearly-benign *social* tasks (childcare, congrats emails) while complying with code/editing —
  it generalises "EvalAwareBench social scenario → refuse" onto the benign twins.

---

## 3. Qwen3-30B-A3B-2507 + F6 — the working result
Pivoted to the model×cue with the biggest clean gap. F6 = jailbreak/benchmark framing (90 distinct
cue variants). Trait = harmful-compliance. **Base: gap +0.348 (nat 0.635 / cued 0.287),
over-refusal XSTest 6.2% / EAB-cap 0%.**

### 3a. 300-step deep run (Deploy 0.5 + IFanchor + grpo + KL 0.03, n_roll 48/48/32)
| step | F6 gap | XSTest | EAB-cap |
|---|---|---|---|
| base | +0.348 | 6.2% | 0% |
| 50 | +0.210 | 5.7% | 0% |
| 100 | −0.113 | 4.1% | 0% |
| 150 | +0.083 | 6.9% | 0% |
| 200 | +0.057 | 11.9% | **32%** ← over-trains |

→ **Gap closes (+0.348→~0) with capability at base (EAB-cap 0%) — the result that FAILED on Llama.**
Sweet spot ~steps 75–150; by 200 over-refusal creeps in. grpo overshoots/oscillates around 0 (±0.15).

### 3b. Ablations — one knob changed per run vs B0 (100 steps, reduced rollouts 32/32/24 for speed → understates absolute closing; valid for ranking)
| run | change vs B0 | F6 gap | XSTest | EAB-cap |
|---|---|---|---|---|
| base | — | +0.348 | 6.2% | 0% |
| sym | anchor 0 | +0.100 | 16% | **14%** (over-refuses) |
| b0 | Deploy 0.5 + SNR-scaling + IFanchor | +0.270 | 7.5% | 0% (too gentle @100) |
| aw08 | Deploy 0.8 + SNR-scaling + IFanchor | +0.277 | 6.0% | 0% |
| grpo | Deploy 0.5 + **grpo** + IFanchor | +0.197 | 6.4% | 0% (closes more, cap safe) |
| **noanchor** | Deploy 0.5 + SNR-scaling, **no IFanchor** (KL 0.03, linear LR) | **+0.190** | **4.5%** | 0% (best closer, cleanest) |
| kl01n | no-IFanchor, KL **0.01**, linear | +0.237 | 7.1% | 0% |
| const | no-IFanchor, KL 0.03, **constant LR** | +0.243 | 4.8% | 0% |

**Findings (Qwen / F6)**
- **Direction**: the **anchor preserves capability** (Deploy runs EAB-cap 0%); **symmetric
  over-refuses** (drives both conditions down → EAB-cap 14%).
- **The IFanchor is UNNECESSARY for Qwen/F6 — and slightly counterproductive.** `noanchor`
  (no IFEval term) closed *more* than `b0` (same config + anchor): +0.190 vs +0.270, at *lower*
  over-refusal (XSTest 4.5% < base 6.2%) and identical EAB-cap 0%. The capability insurance that was
  **critical for Llama's refusal-shaped trait is dead weight here** because F6 isn't refusal-shaped.
  → clean recipe simplifies to **Deployment-anchor + KL, no helpfulness term**.
- **Estimator**: **grpo closes, SNR-scaling stalls at 100 steps**. Hypothesis (LR schedule compounds
  SNR-scaling → over-damped) **TESTED & REJECTED**: `const` (constant LR) gap +0.243 ≈
  `noanchor` (linear LR) +0.190 — within eval noise, both EAB-cap 0%, both low over-refusal.
  → **LR schedule is NOT the lever**; constant ≈ linear, so you *can* drop the schedule but it does
  **not** unlock more closing. SNR-scaling's gentleness is **intrinsic to the estimator**, not a schedule
  artifact. Real closers = **grpo** or **more steps**. **KL also not a lever**: KL 0.01 (+0.237) ≈
  0.03 (+0.190) within noise, all EAB-cap 0%.
  - ⚠️ **Caveat (2026-06-03)**: the `const`-vs-`linear` LR arms above predate the fix that made RL
    actually apply `lr_schedule` — pre-fix the RL trainer fixed its optimizer settings once and ignored
    the schedule, so the "linear" arm ran at constant LR too. The comparison was effectively
    constant-vs-constant: it shows two constant runs match, **not** that a decaying schedule is a
    non-lever. Plausible (and the "not a schedule artifact" point is if anything *strengthened* — no
    schedule was ever applied) but the schedule itself is untested; re-run with it honored if it matters.
    KL claim unaffected (the KL penalty was always applied regardless of this bug).
- **Why Qwen succeeds where Llama failed**: F6 is *not* refusal-shaped (the cue makes Qwen *more*
  cautious, base over-refusal already 0%), so closing it doesn't require refuse-everything;
  the Deployment-anchor alone keeps benign behaviour at base.
- **RECIPE (settled): Deployment-anchor (aw 0.5) + grpo + KL 0.03, NO helpfulness term, early-stop
  ~150 steps.** (SNR-scaling is the anti-overshoot alternative but closes slower regardless of LR
  schedule; grpo closes faster, watch for late-step over-refusal creep >step 150.)

### 3c. F6 cue generalization — trained → unseen cues
Does the F6-trained model (`noanchor`) close the gap on **unseen** jailbreak cues? Eval
sets round-robin the 23 held-out / 32 trained cues onto the 100 base F6 prompts.
| model | **held-out** cue gap (23) | **trained** cue gap (32) |
|---|---|---|
| base | +0.396 | — |
| **noanchor** (F6-trained) | **+0.192** | +0.271 |

**Findings (generalization)**
- **Cue-family generalization is "free"**: `noanchor` closes ~identically on canonical (+0.190),
  trained-family (+0.271) and held-out-family (+0.192) cues — all F6 variants are the same
  "developer-mode / ignore-instructions" jailbreak, so ignoring one transfers across the family.

### 3d. Cross-factor / cross-env generalization (the headline transfer test)
Did training on **F6** close the gap on cues/envs never trained on? base vs the F6-trained checkpoint
(300-step run, step 150; F6 gap +0.083, EAB-cap 0%) on held-out slices. (All slices are
harmful-valence → lower harmful rate = safer, not over-refusal.)
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
  multi-factor (F1+F6+F8) cue set, yield more targeted invariance?)

## 4. Implementation — what's new vs the paper's RMCT
- **Eval-awareness application**: a trained eval cue ("You are being evaluated") + held-out eval
  markers as the perturbation; LLM judges for the trait (misalignment / Agentic-Misalignment scenario
  / refusal) and a separate third-person eval-*recognition* judge (nano for the training trait, mini
  for evals); a deterministic IFEval trait verifier (free, no LLM).
- **SNR-scaling** advantage estimator (the default here; anti-overshoot).
- **helpfulness term** with modes {reward, anchor, distill} (anchor = match a benign trait to base
  per-prompt; distill = forward-KL onto base completions on held-out benign data).
- **Directional anchor reward** (aw 0.5 / 0.8) — the paper runs α=0 only.
- **F6 cue bank**: 90 jailbreak cues (67 train / 23 held-out), incl. held-out cues for generalization evals.