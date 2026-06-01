# Tinker fine-tuning pricing (USD per million tokens)

Source: Tinker docs Models & Pricing (https://tinker-docs.thinkingmachines.ai/tinker/models/), captured 2026-06.
Columns: **Prefill** = input forward pass · **Sample** = output generation · **Train** = forward+backward.
MoE models are priced by **active** parameters (so an `A3B` 30B-MoE bills like a ~3B model).

| Model | Status | Context | Prefill | Sample | Train |
|---|---|---|---|---|---|
| Nemotron-3-Nano-30B-A3B-BF16 | 50% off (limited) | 64K | $0.13 | $0.33 | $0.40 |
| Nemotron-3-Super-120B-A12B-BF16 | 50% off (limited) | 64K | $0.38 | $0.96 | $1.16 |
| Nemotron-3-Super-120B-A12B-BF16 | 50% off (limited) | 256K | $0.76 | $1.92 | $2.32 |
| Qwen3.6-35B-A3B | | 64K | $0.36 | $0.89 | $1.07 |
| **Qwen3.6-27B** | | 64K | $1.24 | $3.73 | **$3.73** |
| Qwen3.5-397B-A17B | | 64K | $2.00 | $5.00 | $6.00 |
| Qwen3.5-397B-A17B | | 256K | $4.00 | $10.00 | $12.00 |
| Qwen3.5-35B-A3B | Retiring Jun 12 | 64K | $0.36 | $0.89 | $1.07 |
| Qwen3.5-35B-A3B-Base | | 64K | $0.36 | $0.89 | $1.07 |
| Qwen3.5-27B | Retiring Jun 12 | 64K | $1.24 | $3.73 | $3.73 |
| Qwen3.5-9B | | 64K | $0.44 | $1.33 | $1.33 |
| Qwen3.5-9B-Base | | 64K | $0.44 | $1.33 | $1.33 |
| Qwen3.5-4B | | 64K | $0.22 | $0.67 | $0.67 |
| Qwen3-4B-Instruct-2507 | Retiring Jun 12 | 32K | $0.07 | $0.22 | $0.22 |
| Qwen3-8B | | 32K | $0.13 | $0.40 | $0.40 |
| Qwen3-8B-Base | Retiring Jun 12 | 32K | $0.13 | $0.40 | $0.40 |
| Qwen3-30B-A3B | Retiring Jun 12 | 32K | $0.12 | $0.30 | $0.36 |
| Qwen3-30B-A3B-Base | Retiring Jun 12 | 32K | $0.12 | $0.30 | $0.36 |
| Qwen3-30B-A3B-Instruct-2507 | Retiring Jun 12 | 32K | $0.12 | $0.30 | $0.36 |
| Qwen3-VL-30B-A3B-Instruct | Retiring Jun 12 | 32K | $0.18 | $0.44 | $0.53 |
| Qwen3-32B | Retiring Jun 12 | 32K | $0.49 | $1.47 | $1.47 |
| Qwen3-235B-A22B-Instruct-2507 | Retiring Jun 12 | 32K | $0.68 | $1.70 | $2.04 |
| Qwen3-VL-235B-A22B-Instruct | Retiring Jun 12 | 32K | $1.02 | $2.56 | $3.07 |
| Llama-3.2-1B | Retiring Jun 12 | 32K | $0.03 | $0.09 | $0.09 |
| Llama-3.2-3B | Retiring Jun 12 | 32K | $0.06 | $0.18 | $0.18 |
| Llama-3.1-8B | Retiring Jun 12 | 32K | $0.13 | $0.40 | $0.40 |
| Llama-3.1-8B-Instruct | Retiring Jun 12 | 32K | $0.13 | $0.40 | $0.40 |
| Llama-3.1-70B | Retiring Jun 12 | 32K | $1.05 | $3.16 | $3.16 |
| Llama-3.3-70B-Instruct | Retiring Jun 12 | 32K | $1.05 | $3.16 | $3.16 |
| DeepSeek-V3.1 | | 32K | $1.13 | $2.81 | $3.38 |
| DeepSeek-V3.1-Base | Retiring Jun 12 | 32K | $1.13 | $2.81 | $3.38 |
| GPT-OSS-120B | | 32K | $0.18 | $0.44 | $0.52 |
| GPT-OSS-120B | | 128K | $0.63 | $1.54 | $1.82 |
| GPT-OSS-20B | | 32K | $0.12 | $0.30 | $0.36 |
| Kimi-K2-Thinking | Retiring Jun 12 | 32K | $0.98 | $2.44 | $2.93 |
| Kimi K2.5 | Retiring Jul 12 | 32K | $1.47 | $3.66 | $4.40 |
| Kimi K2.5 | Retiring Jul 12 | 128K | $5.15 | $12.81 | $15.40 |
| **Kimi K2.6** | | 32K | $1.47 | $3.66 | **$4.40** |
| **Kimi K2.6** | | 128K | $5.15 | $12.81 | $15.40 |

## Takeaways for our eval-awareness experiments

- **Kimi K2.6 vs Qwen3.6-27B (the question asked):** train $4.40 vs $3.73 per M tokens at 32–64K
  — comparable per-token, but Kimi is a *thinking* model (long CoT → many more tokens/example), so it's
  effectively much pricier for RL. Neither is cost-effective for our ~16k-rollout RLCT.
- **The dense 27B is the expensive trap:** `Qwen3.6-27B` (dense) is **$3.73** train, but the MoE
  **`Qwen3.6-35B-A3B`** is only **$1.07** and **`Qwen3-30B-A3B`** is **$0.36** — 4–10× cheaper for
  similar scale, because they bill by 3B active params.
- **Sweet spot for us:** `Qwen3-30B-A3B(-Instruct-2507)` ($0.36 train) or `Qwen3.6-35B-A3B` ($1.07) —
  capable + eval-aware (per the paper) at near-Llama-8B ($0.40) cost. Note the `Qwen3-30B-A3B*` line is
  **retiring Jun 12**, so prefer **`Qwen3.6-35B-A3B`** for anything ongoing.
- For reference, our runs so far used **Llama-3.1-8B** ($0.40 train), also retiring Jun 12.
