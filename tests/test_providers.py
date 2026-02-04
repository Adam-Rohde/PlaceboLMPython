"""Tests for provider-specific placebos."""

import pytest
from placebos import OpenAIPlacebo, AnthropicPlacebo, GenericPlacebo


class TestOpenAIPlacebo:
    """Tests for OpenAI-compatible placebo."""

    def test_create_completion(self):
        placebo = OpenAIPlacebo(
            model="gpt-4",
            default_response="Test completion",
        )
        response = placebo.create_completion("Hello")

        assert response["model"] == "gpt-4"
        assert response["choices"][0]["text"] == "Test completion"
        assert response["choices"][0]["finish_reason"] == "stop"
        assert "usage" in response

    def test_create_chat_completion(self):
        placebo = OpenAIPlacebo(
            model="gpt-3.5-turbo",
            default_response="Chat response",
        )
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        response = placebo.create_chat_completion(messages)

        assert response["model"] == "gpt-3.5-turbo"
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert response["choices"][0]["message"]["content"] == "Chat response"


class TestAnthropicPlacebo:
    """Tests for Anthropic-compatible placebo."""

    def test_create_message(self):
        placebo = AnthropicPlacebo(
            model="claude-3-opus",
            default_response="Claude response",
        )
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        response = placebo.create_message(messages)

        assert response["model"] == "claude-3-opus"
        assert response["role"] == "assistant"
        assert response["content"][0]["type"] == "text"
        assert response["content"][0]["text"] == "Claude response"
        assert response["stop_reason"] == "end_turn"
        assert "usage" in response


class TestGenericPlacebo:
    """Tests for generic customizable placebo."""

    def test_default_formatter(self):
        placebo = GenericPlacebo(default_response="Generic response")
        result = placebo.formatted_generate("Test")

        assert result["content"] == "Generic response"

    def test_custom_formatter(self):
        def custom_format(response):
            return {"result": response.content, "success": True}

        placebo = GenericPlacebo(
            response_formatter=custom_format,
            default_response="Formatted",
        )
        result = placebo.formatted_generate("Test")

        assert result["result"] == "Formatted"
        assert result["success"] is True
