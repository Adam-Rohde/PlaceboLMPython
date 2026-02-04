"""Tests for core placebos functionality."""

import pytest
from placebos import PlaceboLM, PlaceboConfig, PlaceboResponse


class TestPlaceboResponse:
    """Tests for PlaceboResponse class."""

    def test_response_str(self):
        response = PlaceboResponse(content="Hello")
        assert str(response) == "Hello"

    def test_response_to_dict(self):
        response = PlaceboResponse(
            content="Hello",
            model="test-model",
            tokens_used=10,
            latency_ms=50.0,
        )
        result = response.to_dict()
        assert result["content"] == "Hello"
        assert result["model"] == "test-model"
        assert result["tokens_used"] == 10
        assert result["latency_ms"] == 50.0


class TestPlaceboConfig:
    """Tests for PlaceboConfig class."""

    def test_default_config(self):
        config = PlaceboConfig()
        assert config.default_response == "This is a placebo response."
        assert config.latency_ms == 0.0
        assert config.token_count == 10

    def test_custom_config(self):
        config = PlaceboConfig(
            default_response="Custom",
            latency_ms=100,
            token_count=50,
        )
        assert config.default_response == "Custom"
        assert config.latency_ms == 100
        assert config.token_count == 50


class TestPlaceboLM:
    """Tests for PlaceboLM class."""

    def test_default_response(self):
        placebo = PlaceboLM(default_response="Test response")
        response = placebo.generate("Any prompt")
        assert response.content == "Test response"

    def test_response_map(self):
        placebo = PlaceboLM(
            response_map={"Hello": "Hi there!"},
            default_response="Default",
        )
        assert placebo.generate("Hello").content == "Hi there!"
        assert placebo.generate("Other").content == "Default"

    def test_response_func(self):
        def echo(prompt):
            return f"Echo: {prompt}"

        placebo = PlaceboLM(response_func=echo)
        assert placebo.generate("Test").content == "Echo: Test"

    def test_chat_interface(self):
        placebo = PlaceboLM(default_response="Chat response")
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        response = placebo.chat(messages)
        assert response.content == "Chat response"

    def test_call_history(self):
        placebo = PlaceboLM(default_response="Response")
        placebo.generate("Prompt 1")
        placebo.generate("Prompt 2")

        history = placebo.call_history
        assert len(history) == 2
        assert history[0]["prompt"] == "Prompt 1"
        assert history[1]["prompt"] == "Prompt 2"

    def test_clear_history(self):
        placebo = PlaceboLM(default_response="Response")
        placebo.generate("Test")
        placebo.clear_history()
        assert len(placebo.call_history) == 0

    def test_assert_called(self):
        placebo = PlaceboLM(default_response="Response")

        with pytest.raises(AssertionError):
            placebo.assert_called()

        placebo.generate("Test")
        placebo.assert_called()  # Should not raise

    def test_assert_called_with(self):
        placebo = PlaceboLM(default_response="Response")
        placebo.generate("Specific prompt")

        placebo.assert_called_with("Specific prompt")

        with pytest.raises(AssertionError):
            placebo.assert_called_with("Different prompt")

    def test_assert_call_count(self):
        placebo = PlaceboLM(default_response="Response")
        placebo.generate("1")
        placebo.generate("2")
        placebo.generate("3")

        placebo.assert_call_count(3)

        with pytest.raises(AssertionError):
            placebo.assert_call_count(5)

    def test_config_integration(self):
        config = PlaceboConfig(
            default_response="Config response",
            token_count=25,
        )
        placebo = PlaceboLM(config=config)
        response = placebo.generate("Test")

        assert response.content == "Config response"
        assert response.tokens_used == 25
