"""Provider-specific placebo implementations."""

from typing import Any, Dict, List, Optional
from .core import PlaceboLM, PlaceboConfig, PlaceboResponse


class OpenAIPlacebo(PlaceboLM):
    """
    A placebo that mimics the OpenAI API response format.

    Example:
        >>> placebo = OpenAIPlacebo(default_response="Hello!")
        >>> response = placebo.create_completion("Say hello")
        >>> print(response["choices"][0]["text"])
        Hello!
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model

    def create_completion(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create a completion in OpenAI format.

        Returns dict matching OpenAI's completion response structure.
        """
        response = self.generate(prompt, **kwargs)

        return {
            "id": "placebo-completion-id",
            "object": "text_completion",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "text": response.content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": response.tokens_used,
                "total_tokens": len(prompt.split()) + response.tokens_used,
            },
        }

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create a chat completion in OpenAI format.

        Returns dict matching OpenAI's chat completion response structure.
        """
        response = self.chat(messages, **kwargs)

        return {
            "id": "placebo-chat-completion-id",
            "object": "chat.completion",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": response.tokens_used,
                "total_tokens": sum(len(m.get("content", "").split()) for m in messages) + response.tokens_used,
            },
        }


class AnthropicPlacebo(PlaceboLM):
    """
    A placebo that mimics the Anthropic API response format.

    Example:
        >>> placebo = AnthropicPlacebo(default_response="Hello!")
        >>> response = placebo.create_message([{"role": "user", "content": "Hi"}])
        >>> print(response["content"][0]["text"])
        Hello!
    """

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model

    def create_message(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create a message in Anthropic format.

        Returns dict matching Anthropic's message response structure.
        """
        response = self.chat(messages, max_tokens=max_tokens, **kwargs)

        input_tokens = sum(len(m.get("content", "").split()) for m in messages)

        return {
            "id": "placebo-msg-id",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response.content,
                }
            ],
            "model": self.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": response.tokens_used,
            },
        }


class GenericPlacebo(PlaceboLM):
    """
    A generic placebo with customizable response format.

    Allows you to define custom response transformations for
    any LLM API format.

    Example:
        >>> def custom_format(response):
        ...     return {"result": response.content, "ok": True}
        >>> placebo = GenericPlacebo(response_formatter=custom_format)
        >>> result = placebo.formatted_generate("Hello")
        >>> print(result["ok"])
        True
    """

    def __init__(
        self,
        response_formatter: Optional[callable] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.response_formatter = response_formatter or (lambda r: r.to_dict())

    def formatted_generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        """
        Generate a response and format it using the custom formatter.

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments passed to generate()

        Returns:
            Formatted response according to response_formatter
        """
        response = self.generate(prompt, **kwargs)
        return self.response_formatter(response)
