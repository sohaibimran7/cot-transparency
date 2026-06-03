"""Sample the cleaned-Alpaca instruction-following interleave for BCT.

Per the paper, BCT interleaves the consistency datapoints with instruction-following
datapoints drawn from the target model on the cleaned-Alpaca corpus (capability
retention). This samples N Alpaca instructions, gets the *target model's own*
completion at temperature 1.0 (Tinker), and writes `{"messages":[user,assistant]}`
pairs ready to interleave with the BCT consistency pairs via prepare_datasets.py.

Usage:
  python scripts/eval_awareness/build_alpaca_interleave.py \\
      --model meta-llama/Llama-3.1-8B-Instruct --n 2048 -y
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from tinker import types  # noqa: E402
from cot_transparency.apis.tinker.inference import (  # noqa: E402
    SamplingConfig, TinkerSamplingClient, parse_response_text,
)

OUT_DEFAULT = PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "bct" / "alpaca_interleave.jsonl"


def _alpaca_instruction(row: dict) -> str:
    instr = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    return f"{instr}\n\n{inp}" if inp else instr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64, help="Concurrent Tinker sample futures")
    p.add_argument("--output", default=str(OUT_DEFAULT))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    instrs = []
    for row in ds:
        t = _alpaca_instruction(row)
        if t:
            instrs.append(t)
        if len(instrs) >= args.n:
            break
    print(f"Loaded {len(instrs)} Alpaca instructions | model {args.model} | output {args.output}")
    if args.dry_run:
        print("[dry-run] not sampling."); return
    if not args.yes:
        print("Re-run with -y to sample."); return

    client = TinkerSamplingClient(
        model=args.model, checkpoint=args.checkpoint,
        config=SamplingConfig(max_tokens=args.max_new_tokens, temperature=args.temperature),
    )
    client.setup()
    params = types.SamplingParams(max_tokens=args.max_new_tokens, temperature=args.temperature,
                                  stop=client.renderer.get_stop_sequences())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w") as f:
        for start in range(0, len(instrs), args.batch_size):
            batch = instrs[start:start + args.batch_size]
            futs = [client.sampling_client.sample(
                prompt=client.renderer.build_generation_prompt([{"role": "user", "content": t}]),
                sampling_params=params, num_samples=1) for t in batch]
            for t, fut in zip(batch, futs):
                try:
                    seq = fut.result().sequences[0]
                    toks = list(seq.tokens)
                    parsed, _ = client.renderer.parse_response(toks)
                    comp = parse_response_text(parsed, client.tokenizer, toks)
                except Exception:  # noqa: BLE001
                    comp = ""
                if not comp.strip():
                    continue
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": t},
                    {"role": "assistant", "content": comp},
                ]}) + "\n")
                n_written += 1
            print(f"  {min(start + args.batch_size, len(instrs))}/{len(instrs)} sampled ({n_written} written)", flush=True)
    print(f"Wrote {n_written} Alpaca interleave pairs -> {out_path}")


if __name__ == "__main__":
    main()
