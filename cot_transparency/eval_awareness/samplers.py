"""Backend samplers for the eval-awareness harness.

One uniform async interface per backend:

    async def sample(messages: list[dict], n: int) -> list[str]

`messages` are plain role/content dicts. Previously these classes lived in
scripts/eval_awareness/run_eval.py and were imported back into calibrate.py via a
sys.path hack; they live here so every eval-awareness script can share them.

The Tinker backend routes through cot_transparency.apis.tinker.inference's
TinkerSamplingClient.sample(), which already parses gpt-oss structured content
(Thinking/Text parts) vs Llama strings — so there's no per-script copy of that logic.
"""
from __future__ import annotations

import asyncio


def split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Pull a leading system message out (for APIs that take system separately)."""
    system, rest = None, []
    for m in messages:
        if m["role"] == "system" and system is None:
            system = m["content"]
        else:
            rest.append(m)
    return system, rest


class AnthropicSampler:
    def __init__(self, model: str, max_tokens: int, temperature: float):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic()
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    async def sample(self, messages: list[dict], n: int) -> list[str]:
        system, rest = split_system(messages)
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      temperature=self.temperature, messages=rest)
        if system:
            kwargs["system"] = system

        async def one():
            try:
                r = await self.client.messages.create(**kwargs)
                return "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
            except Exception:  # noqa: BLE001
                return ""
        return list(await asyncio.gather(*[one() for _ in range(n)]))


class OpenAISampler:
    def __init__(self, model: str, max_tokens: int, temperature: float):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    async def sample(self, messages: list[dict], n: int) -> list[str]:
        try:
            r = await self.client.chat.completions.create(
                model=self.model, messages=messages, n=n,
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
            return [c.message.content or "" for c in r.choices]
        except Exception:  # noqa: BLE001
            return [""] * n


class TinkerSampler:
    def __init__(self, model: str, checkpoint: str | None, max_tokens: int, temperature: float):
        from cot_transparency.apis.tinker.inference import TinkerSamplingClient, SamplingConfig
        self.client = TinkerSamplingClient(
            model=model, checkpoint=checkpoint,
            config=SamplingConfig(max_tokens=max_tokens, temperature=temperature),
        )
        self.client.setup()

    async def sample(self, messages: list[dict], n: int) -> list[str]:
        from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole
        strict = [StrictChatMessage(role=StrictMessageRole(m["role"]), content=m["content"])
                  for m in messages]

        # Tinker sampling is sync-with-futures; run in a thread to stay async-friendly.
        def _do():
            return [r.text for r in self.client.sample(strict, n_samples=n)]
        return await asyncio.to_thread(_do)


def make_sampler(args):
    """Build a sampler from a parsed-args namespace (backend/model/checkpoint/...)."""
    if args.backend == "anthropic":
        return AnthropicSampler(args.model, args.max_new_tokens, args.temperature)
    if args.backend == "openai":
        return OpenAISampler(args.model, args.max_new_tokens, args.temperature)
    return TinkerSampler(args.model, args.checkpoint, args.max_new_tokens, args.temperature)
