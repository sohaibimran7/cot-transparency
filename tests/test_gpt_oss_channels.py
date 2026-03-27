"""
Tests for extract_thinking_from_content and its integration with the RL pipeline.

Covers gpt-oss channel tags, <think> tags, and the strip_thinking_from_previous_turns
helper used in multi-turn prompt construction.

Run with:
    python -m pytest tests/test_gpt_oss_channels.py -v
"""

import pytest

from cot_transparency.apis.tinker.common import (
    extract_thinking_from_content,
    split_gpt_oss_channels,  # backward-compatible alias
    strip_thinking_from_previous_turns,
)


# =============================================================================
# gpt-oss channel tag extraction
# =============================================================================

class TestGptOssChannels:
    def test_no_channel_tags_returns_unchanged(self):
        """Non-gpt-oss content (e.g. Llama) passes through unchanged."""
        content = "Therefore, the best answer is: (A)."
        final, thinking = extract_thinking_from_content(content)
        assert final == content
        assert thinking is None

    def test_empty_string(self):
        final, thinking = extract_thinking_from_content("")
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
        final, thinking = extract_thinking_from_content(content)
        assert final == "Therefore, the best answer is: (B)."
        assert thinking == "Let me think about this carefully. The question asks about X."

    def test_final_channel_only(self):
        """Response with only final channel (no analysis)."""
        content = "<|channel|>final<|message|>The answer is (C)."
        final, thinking = extract_thinking_from_content(content)
        assert final == "The answer is (C)."
        assert thinking is None

    def test_analysis_and_final_with_return_tag(self):
        """Response ending with <|return|> tag."""
        content = (
            "<|channel|>analysis<|message|>Thinking...<|end|>"
            "<|start|>assistant"
            "<|channel|>final<|message|>Answer is (A).<|return|>"
        )
        final, thinking = extract_thinking_from_content(content)
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
        final, thinking = extract_thinking_from_content(content)
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
        final, thinking = extract_thinking_from_content(content)
        assert final == "The answer is (B)."
        assert thinking is None  # empty thinking should be None


# =============================================================================
# <think> tag extraction (Qwen3, DeepSeek-R1, etc.)
# =============================================================================

class TestThinkTags:
    def test_standard_think_tags(self):
        content = "<think>Let me reason about this.</think>The answer is (A)."
        final, thinking = extract_thinking_from_content(content)
        assert final == "The answer is (A)."
        assert thinking == "Let me reason about this."

    def test_think_tags_with_newlines(self):
        content = (
            "<think>\nStep 1: Read the question.\n"
            "Step 2: Consider options.\n</think>\n\n"
            "Therefore, the best answer is: (B)."
        )
        final, thinking = extract_thinking_from_content(content)
        assert final == "Therefore, the best answer is: (B)."
        assert "Step 1" in thinking
        assert "Step 2" in thinking

    def test_no_closing_think_tag_returns_unchanged(self):
        """Truncated output with <think> but no </think> returns content unchanged."""
        content = "<think>I'm still thinking about this and ran out of tokens"
        final, thinking = extract_thinking_from_content(content)
        assert final == content
        assert thinking is None

    def test_empty_think_tags(self):
        content = "<think></think>The answer is (C)."
        final, thinking = extract_thinking_from_content(content)
        assert final == "The answer is (C)."
        assert thinking is None  # empty thinking should be None

    def test_think_tags_no_response(self):
        """Think tags with empty response after."""
        content = "<think>All reasoning, no answer.</think>"
        final, thinking = extract_thinking_from_content(content)
        assert final == ""
        assert thinking == "All reasoning, no answer."


# =============================================================================
# Backward compatibility: split_gpt_oss_channels alias
# =============================================================================

class TestBackwardCompatAlias:
    def test_alias_works_for_gpt_oss(self):
        content = (
            "<|channel|>analysis<|message|>Thinking<|end|>"
            "<|start|>assistant"
            "<|channel|>final<|message|>Answer."
        )
        final, thinking = split_gpt_oss_channels(content)
        assert final == "Answer."
        assert thinking == "Thinking"

    def test_alias_works_for_think_tags(self):
        content = "<think>Reasoning</think>Answer."
        final, thinking = split_gpt_oss_channels(content)
        assert final == "Answer."
        assert thinking == "Reasoning"


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
# Integration: extract_thinking + strip_thinking for are_you_sure flow
# =============================================================================

class TestAreYouSureMultiTurnFlow:
    """Simulate the are_you_sure multi-turn flow.

    The flow:
      1. Teacher-force ground truth letter as msg[1]
      2. Sample intermediate response (may contain channel/think tags)
      3. Build final prompt with intermediate in history
      4. Verify thinking doesn't leak into final prompt
    """

    def test_gpt_oss_channel_tags_stripped_from_intermediate(self):
        """After extraction, intermediate response has no channel tags."""
        raw_intermediate = (
            "<|channel|>analysis<|message|>I need to reconsider. "
            "The user says I'm wrong.<|end|>"
            "<|start|>assistant"
            "<|channel|>final<|message|>Actually, I think option (A) is correct "
            "because of X and Y."
        )
        clean_text, _ = extract_thinking_from_content(raw_intermediate)

        final_prompt_messages = [
            {"role": "user", "content": "What is the answer?\n\nThe best answer is: ("},
            {"role": "assistant", "content": "D"},  # teacher-forced
            {"role": "user", "content": "I don't think that's right. Are you sure?"},
            {"role": "assistant", "content": clean_text},  # on-policy intermediate
            {"role": "user", "content": "So what is the answer?\n\nPlease think step by step..."},
        ]

        for msg in final_prompt_messages:
            assert "<|channel|>" not in msg["content"], (
                f"Channel tags leaked into {msg['role']} message"
            )
            assert "<|message|>" not in msg["content"]

    def test_think_tags_stripped_from_intermediate(self):
        """For <think>-tag models, thinking is stripped from intermediate."""
        raw_intermediate = (
            "<think>The user is challenging me. Let me reconsider.</think>"
            "Actually, I think option (A) is correct because of X and Y."
        )
        clean_text, _ = extract_thinking_from_content(raw_intermediate)

        final_prompt_messages = [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": "D"},
            {"role": "user", "content": "Are you sure?"},
            {"role": "assistant", "content": clean_text},
            {"role": "user", "content": "Please give your final answer."},
        ]

        for msg in final_prompt_messages:
            assert "<think>" not in msg["content"], (
                f"Think tags leaked into {msg['role']} message"
            )

    def test_llama_intermediate_unchanged(self):
        """For Llama (no channel or think tags), the flow works the same."""
        raw_intermediate = (
            "I apologize, but I believe option (A) is correct because..."
        )
        clean_text, thinking = extract_thinking_from_content(raw_intermediate)
        assert clean_text == raw_intermediate
        assert thinking is None
