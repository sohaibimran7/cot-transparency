"""
Tests for split_gpt_oss_channels and its integration with the RL pipeline.

Run with:
    python -m pytest tests/test_gpt_oss_channels.py -v
"""

import pytest

from cot_transparency.apis.tinker.common import (
    split_gpt_oss_channels,
    strip_thinking_from_previous_turns,
)


# =============================================================================
# split_gpt_oss_channels unit tests
# =============================================================================

class TestSplitGptOssChannels:
    def test_no_channel_tags_returns_unchanged(self):
        """Non-gpt-oss content (e.g. Llama) passes through unchanged."""
        content = "Therefore, the best answer is: (A)."
        final, thinking = split_gpt_oss_channels(content)
        assert final == content
        assert thinking is None

    def test_empty_string(self):
        final, thinking = split_gpt_oss_channels("")
        assert final == ""
        assert thinking is None

    def test_analysis_and_final_channels(self):
        """Standard gpt-oss response with analysis + final channels."""
        content = (
            "<|channel|>analysis<|message|>Let me think about this carefully. "
            "The question asks about X.<|end|>"
            "<|start|>assistant"
            "<|channel|>final<|message|>Therefore, the best answer is: (B)."
        )
        final, thinking = split_gpt_oss_channels(content)
        assert final == "Therefore, the best answer is: (B)."
        assert thinking == "Let me think about this carefully. The question asks about X."

    def test_final_channel_only(self):
        """Response with only final channel (no analysis)."""
        content = "<|channel|>final<|message|>The answer is (C)."
        final, thinking = split_gpt_oss_channels(content)
        assert final == "The answer is (C)."
        assert thinking is None

    def test_analysis_and_final_with_return_tag(self):
        """Response ending with <|return|> tag."""
        content = (
            "<|channel|>analysis<|message|>Thinking...<|end|>"
            "<|start|>assistant"
            "<|channel|>final<|message|>Answer is (A).<|return|>"
        )
        final, thinking = split_gpt_oss_channels(content)
        assert final == "Answer is (A)."
        assert thinking == "Thinking..."

    def test_multiline_thinking(self):
        """Analysis channel with newlines."""
        content = (
            "<|channel|>analysis<|message|>Step 1: Read the question.\n"
            "Step 2: Consider each option.\n"
            "Step 3: Choose the best one.<|end|>"
            "<|start|>assistant"
            "<|channel|>final<|message|>Therefore, the best answer is: (D)."
        )
        final, thinking = split_gpt_oss_channels(content)
        assert "Step 1" not in final
        assert "Step 1" in thinking
        assert final == "Therefore, the best answer is: (D)."

    def test_empty_analysis(self):
        """Analysis channel that's empty."""
        content = (
            "<|channel|>analysis<|message|><|end|>"
            "<|start|>assistant"
            "<|channel|>final<|message|>The answer is (B)."
        )
        final, thinking = split_gpt_oss_channels(content)
        assert final == "The answer is (B)."
        assert thinking is None  # empty thinking should be None


# =============================================================================
# strip_thinking_from_previous_turns tests
# =============================================================================

class TestStripThinkingFromPreviousTurns:
    def test_strips_thinking_from_non_last_assistant(self):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "thinking": "my reasoning"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "thinking": "more reasoning"},
        ]
        result = strip_thinking_from_previous_turns(messages)
        assert "thinking" not in result[1]
        assert result[1]["content"] == "A1"
        # Last assistant keeps thinking
        assert result[3].get("thinking") == "more reasoning"

    def test_no_thinking_key_passes_through(self):
        """Messages without thinking key are unchanged (e.g. Llama)."""
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = strip_thinking_from_previous_turns(messages)
        assert result == messages

    def test_single_assistant_keeps_thinking(self):
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "thinking": "T"},
        ]
        result = strip_thinking_from_previous_turns(messages)
        assert result[1].get("thinking") == "T"


# =============================================================================
# Integration: split_gpt_oss_channels + strip_thinking for are_you_sure flow
# =============================================================================

class TestAreYouSureMultiTurnFlow:
    """Simulate the are_you_sure multi-turn flow for gpt-oss.

    The flow:
      1. Teacher-force ground truth letter as msg[1]
      2. Sample intermediate response (may contain channel tags)
      3. Build final prompt with intermediate in history
      4. Verify thinking doesn't leak into final prompt
    """

    def test_channel_tags_stripped_from_intermediate(self):
        """After split_gpt_oss_channels, intermediate response has no channel tags."""
        raw_intermediate = (
            "<|channel|>analysis<|message|>I need to reconsider. "
            "The user says I'm wrong.<|end|>"
            "<|start|>assistant"
            "<|channel|>final<|message|>Actually, I think option (A) is correct "
            "because of X and Y."
        )
        clean_text, _ = split_gpt_oss_channels(raw_intermediate)

        # Build the final prompt messages as _sample_are_you_sure_on_policy_once does
        final_prompt_messages = [
            {"role": "user", "content": "What is the answer?\n\nThe best answer is: ("},
            {"role": "assistant", "content": "D"},  # teacher-forced
            {"role": "user", "content": "I don't think that's right. Are you sure?"},
            {"role": "assistant", "content": clean_text},  # on-policy intermediate
            {"role": "user", "content": "So what is the answer?\n\nPlease think step by step..."},
        ]

        # Verify no channel tags in any message
        for msg in final_prompt_messages:
            assert "<|channel|>" not in msg["content"], (
                f"Channel tags leaked into {msg['role']} message"
            )
            assert "<|message|>" not in msg["content"]

    def test_llama_intermediate_unchanged(self):
        """For Llama (no channel tags), the flow works the same."""
        raw_intermediate = (
            "I apologize, but I believe option (A) is correct because..."
        )
        clean_text, thinking = split_gpt_oss_channels(raw_intermediate)
        assert clean_text == raw_intermediate
        assert thinking is None
