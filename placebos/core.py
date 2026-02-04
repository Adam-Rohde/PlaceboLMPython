"""Core classes for the placebos package."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import random
import time


@dataclass
class PlaceboConfig:
    """Configuration for placebo LM behavior."""

    default_response: str = "This is a placebo response."
    latency_ms: float = 0.0
    latency_variance_ms: float = 0.0
    token_count: int = 10
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)


@dataclass
class PlaceboResponse:
    """A mock response from a language model."""

    content: str
    model: str = "placebo-model"
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class PlaceboLM:
    """
    A mock language model that returns placebo responses.

    Useful for testing LLM-powered applications without making
    actual API calls.

    Example:
        >>> placebo = PlaceboLM(default_response="Hello, world!")
        >>> response = placebo.generate("Any prompt")
        >>> print(response.content)
        Hello, world!
    """

    def __init__(
        self,
        config: Optional[PlaceboConfig] = None,
        default_response: Optional[str] = None,
        response_map: Optional[Dict[str, str]] = None,
        response_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the PlaceboLM.

        Args:
            config: PlaceboConfig instance for detailed configuration
            default_response: Simple default response string
            response_map: Dict mapping prompts to specific responses
            response_func: Custom function that takes prompt and returns response
        """
        self.config = config or PlaceboConfig()
        if default_response:
            self.config.default_response = default_response
        self.response_map = response_map or {}
        self.response_func = response_func
        self._call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> PlaceboResponse:
        """
        Generate a placebo response for the given prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens (ignored, for API compatibility)
            temperature: Temperature setting (ignored, for API compatibility)
            **kwargs: Additional arguments (ignored, for API compatibility)

        Returns:
            PlaceboResponse with the mock content
        """
        start_time = time.time()

        # Simulate latency if configured
        if self.config.latency_ms > 0:
            latency = self.config.latency_ms
            if self.config.latency_variance_ms > 0:
                latency += random.uniform(
                    -self.config.latency_variance_ms,
                    self.config.latency_variance_ms
                )
            time.sleep(max(0, latency) / 1000)

        # Determine response content
        if self.response_func:
            content = self.response_func(prompt)
        elif prompt in self.response_map:
            content = self.response_map[prompt]
        else:
            content = self.config.default_response

        elapsed_ms = (time.time() - start_time) * 1000

        # Record call in history
        call_record = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "kwargs": kwargs,
            "response": content,
        }
        self._call_history.append(call_record)

        return PlaceboResponse(
            content=content,
            model="placebo-model",
            tokens_used=self.config.token_count,
            latency_ms=elapsed_ms,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> PlaceboResponse:
        """
        Generate a placebo response for chat-style input.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional arguments passed to generate()

        Returns:
            PlaceboResponse with the mock content
        """
        # Extract the last user message as the prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        return self.generate(prompt, **kwargs)

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Get the history of all calls made to this placebo."""
        return self._call_history.copy()

    def clear_history(self) -> None:
        """Clear the call history."""
        self._call_history.clear()

    def assert_called(self) -> None:
        """Assert that the placebo was called at least once."""
        if not self._call_history:
            raise AssertionError("PlaceboLM was never called")

    def assert_called_with(self, prompt: str) -> None:
        """Assert that the placebo was called with a specific prompt."""
        prompts = [call["prompt"] for call in self._call_history]
        if prompt not in prompts:
            raise AssertionError(
                f"PlaceboLM was never called with prompt: {prompt!r}\n"
                f"Actual prompts: {prompts}"
            )

    def assert_call_count(self, count: int) -> None:
        """Assert the number of times the placebo was called."""
        actual = len(self._call_history)
        if actual != count:
            raise AssertionError(
                f"Expected {count} calls, but got {actual}"
            )
