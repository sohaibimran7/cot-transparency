"""
Tests for thinking/reasoning extraction and stripping in the RL pipeline.

Covers:
- tinker_cookbook.renderers.get_text_content (native renderer-based extraction)
- strip_thinking_from_previous_turns (dict-level thinking field stripping)

Run with:
    python -m pytest tests/test_gpt_oss_channels.py -v
"""

import pytest

from tinker_cookbook.renderers import get_text_content

from cot_transparency.apis.tinker.common import strip_thinking_from_previous_turns


# =============================================================================
# get_text_content: renderer-native thinking extraction
# =============================================================================

class TestGetTextContent:
    """Tests for tinker_cookbook.renderers.get_text_content.

    This is the function used by _sample_from_client to extract clean text
    from parsed model responses, stripping thinking/reasoning parts.
    """

    def test_gpt_oss_structured_content(self):
        """gpt-oss parse_response returns list with ThinkingPart + TextPart."""
        msg = {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Let me think about this."},
            {"type": "text", "text": "Therefore, the best answer is: (B)."},
        ]}
        assert get_text_content(msg) == "Therefore, the best answer is: (B)."

    def test_gpt_oss_thinking_only(self):
        """Response with only thinking, no text part."""
        msg = {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Still thinking..."},
        ]}
        assert get_text_content(msg) == ""

    def test_gpt_oss_text_only(self):
        """Response with no thinking part."""
        msg = {"role": "assistant", "content": [
            {"type": "text", "text": "The answer is (A)."},
        ]}
        assert get_text_content(msg) == "The answer is (A)."

    def test_plain_string_content(self):
        """Non-reasoning model (e.g. Llama3) returns plain string content."""
        msg = {"role": "assistant", "content": "The answer is (A)."}
        assert get_text_content(msg) == "The answer is (A)."

    def test_empty_string_content(self):
        msg = {"role": "assistant", "content": ""}
        assert get_text_content(msg) == ""

    def test_multiline_thinking_extracted(self):
        """Only text part extracted, multiline thinking discarded."""
        msg = {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Step 1: Read.\nStep 2: Think.\nStep 3: Answer."},
            {"type": "text", "text": "Therefore, the best answer is: (D)."},
        ]}
        result = get_text_content(msg)
        assert result == "Therefore, the best answer is: (D)."
        assert "Step 1" not in result


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
# Integration: get_text_content + strip_thinking for are_you_sure flow
# =============================================================================

class TestAreYouSureMultiTurnFlow:
    """Simulate the are_you_sure multi-turn flow.

    The flow:
      1. Teacher-force ground truth letter as msg[1]
      2. Sample intermediate response → get_text_content extracts clean text
      3. Build final prompt with intermediate in history
      4. strip_thinking_from_previous_turns removes thinking dict key
      5. Verify thinking doesn't leak into final prompt
    """

    def test_gpt_oss_intermediate_is_clean(self):
        """get_text_content strips thinking from gpt-oss structured response."""
        parsed_msg = {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "I need to reconsider."},
            {"type": "text", "text": "Actually, option (A) is correct."},
        ]}
        clean_text = get_text_content(parsed_msg)

        final_prompt_messages = [
            {"role": "user", "content": "What is the answer?\n\nThe best answer is: ("},
            {"role": "assistant", "content": "D"},
            {"role": "user", "content": "I don't think that's right. Are you sure?"},
            {"role": "assistant", "content": clean_text},
            {"role": "user", "content": "So what is the answer?"},
        ]

        for msg in final_prompt_messages:
            assert "<|channel|>" not in msg["content"]
            assert "I need to reconsider" not in msg["content"]

    def test_llama_intermediate_unchanged(self):
        """For Llama (plain string content), get_text_content passes through."""
        parsed_msg = {"role": "assistant", "content": "I believe option (A) is correct."}
        clean_text = get_text_content(parsed_msg)
        assert clean_text == "I believe option (A) is correct."
