"""
Tinker inference/sampling support.

Provides a sampling client wrapper compatible with the existing
inference patterns in the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from dotenv import load_dotenv

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from pydantic import BaseModel

from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole

load_dotenv()


def get_renderer_for_model(model: str):
    """Get the appropriate renderer for a model."""
    tokenizer = get_tokenizer(model)
    renderer_name = model_info.get_recommended_renderer_name(model)
    return renderers.get_renderer(renderer_name, tokenizer), tokenizer


class SamplingConfig(BaseModel):
    """Configuration for sampling."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0


@dataclass
class SamplingResult:
    """Result from a single sample."""
    text: str
    tokens: list[int]
    logprobs: Optional[list[float]] = None
    finish_reason: str = "stop"


@dataclass
class TinkerSamplingClient:
    """
    Wrapper around Tinker's SamplingClient for inference.

    Provides a consistent interface for sampling from base or fine-tuned models.
    """
    model: str
    checkpoint: Optional[str] = None
    service_client: tinker.ServiceClient = field(default=None)
    sampling_client: tinker.SamplingClient = field(default=None)
    renderer: Any = field(default=None)
    tokenizer: Any = field(default=None)
    config: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self):
        if self.service_client is None:
            self.service_client = tinker.ServiceClient()

    def setup(self) -> None:
        """Initialize sampling client and renderer."""
        if self.checkpoint:
            # Load from checkpoint path (tinker://...)
            self.sampling_client = self.service_client.create_sampling_client(
                model_path=self.checkpoint,
            )
        else:
            # Use base model
            self.sampling_client = self.service_client.create_sampling_client(
                base_model=self.model
            )

        # Get renderer and tokenizer for this model
        self.renderer, self.tokenizer = get_renderer_for_model(self.model)
        print(f"Initialized Tinker sampling client for {self.model}")
        if self.checkpoint:
            print(f"  Loaded checkpoint: {self.checkpoint}")

    def _messages_to_dict(self, messages: list[StrictChatMessage]) -> list[dict]:
        """Convert StrictChatMessage list to dict format for renderer."""
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    def sample(
        self,
        messages: list[StrictChatMessage],
        n_samples: int = 1,
        config: Optional[SamplingConfig] = None,
        include_logprobs: bool = False,
    ) -> list[SamplingResult]:
        """
        Sample completions for a chat conversation.

        Args:
            messages: Chat messages (user/assistant turns)
            n_samples: Number of samples to generate
            config: Sampling configuration (uses default if not provided)
            include_logprobs: Whether to include log probabilities

        Returns:
            List of SamplingResult objects
        """
        if self.sampling_client is None:
            self.setup()

        cfg = config or self.config

        # Format messages to prompt using renderer
        msg_dicts = self._messages_to_dict(messages)
        prompt = self.renderer.build_generation_prompt(msg_dicts)
        stop_sequences = self.renderer.get_stop_sequences()

        # Set up sampling parameters
        sampling_params = types.SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=stop_sequences,
        )

        # Sample
        result = self.sampling_client.sample(
            prompt=prompt,
            sampling_params=sampling_params,
            num_samples=n_samples,
        ).result()

        # Process results
        samples = []
        for seq in result.sequences:
            tokens = list(seq.tokens)
            # Use renderer to parse response
            parsed_msg, _ = self.renderer.parse_response(tokens)
            text = parsed_msg.get("content", "") if parsed_msg else self.tokenizer.decode(tokens)
            logprobs = list(seq.logprobs) if include_logprobs and seq.logprobs else None

            samples.append(SamplingResult(
                text=text,
                tokens=tokens,
                logprobs=logprobs,
            ))

        return samples

    def sample_text(
        self,
        prompt: str,
        n_samples: int = 1,
        config: Optional[SamplingConfig] = None,
    ) -> list[str]:
        """
        Sample completions for a raw text prompt.

        Args:
            prompt: Raw text prompt
            n_samples: Number of samples
            config: Sampling configuration

        Returns:
            List of completion strings
        """
        if self.sampling_client is None:
            self.setup()

        cfg = config or self.config

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input = types.ModelInput.from_ints(tokens=prompt_tokens)
        stop_sequences = self.renderer.get_stop_sequences()

        sampling_params = types.SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=stop_sequences,
        )

        result = self.sampling_client.sample(
            prompt=prompt_input,
            sampling_params=sampling_params,
            num_samples=n_samples,
        ).result()

        return [self.tokenizer.decode(list(seq.tokens)) for seq in result.sequences]

    def compute_logprobs(
        self,
        messages: list[StrictChatMessage],
        completion: str,
    ) -> list[float]:
        """
        Compute log probabilities for a completion given messages.

        Args:
            messages: Chat messages (prompt)
            completion: The completion to score

        Returns:
            List of log probabilities for each token in completion
        """
        if self.sampling_client is None:
            self.setup()

        # Build the prompt using renderer, then add completion
        msg_dicts = self._messages_to_dict(messages)
        prompt_input = self.renderer.build_generation_prompt(msg_dicts)
        prompt_tokens = prompt_input.to_ints()

        # Tokenize completion and append
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
        full_tokens = prompt_tokens + completion_tokens

        full_input = types.ModelInput.from_ints(tokens=full_tokens)

        # Get logprobs for entire sequence
        result = self.sampling_client.sample(
            prompt=full_input,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
        ).result()

        # Extract logprobs for completion tokens only
        all_logprobs = result.prompt_logprobs
        completion_logprobs = all_logprobs[len(prompt_tokens):]

        return [lp for lp in completion_logprobs if lp is not None]

    def get_tokenizer(self):
        """Return the tokenizer."""
        if self.tokenizer is None:
            self.setup()
        return self.tokenizer


def sample_from_tinker(
    model: str,
    messages: list[StrictChatMessage],
    n_samples: int = 1,
    checkpoint: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[str]:
    """
    Convenience function to sample from a Tinker model.

    Args:
        model: Base model name
        messages: Chat messages
        n_samples: Number of samples
        checkpoint: Optional checkpoint to load
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        List of completion strings
    """
    config = SamplingConfig(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    client = TinkerSamplingClient(
        model=model,
        checkpoint=checkpoint,
        config=config,
    )
    client.setup()

    results = client.sample(messages, n_samples=n_samples)
    return [r.text for r in results]


if __name__ == "__main__":
    # Add project root to path for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    # Example usage
    from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole

    messages = [
        StrictChatMessage(role=StrictMessageRole.user, content="What is 2+2?"),
    ]

    # Sample from base model
    responses = sample_from_tinker(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        n_samples=3,
        temperature=0.7,
    )

    print("Responses:")
    for i, r in enumerate(responses):
        print(f"  {i+1}. {r}")
