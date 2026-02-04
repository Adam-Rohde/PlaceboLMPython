# Placebos

A lightweight Python package for creating mock/placebo responses from language models. Useful for testing LLM-powered applications without incurring API costs.

## Installation

```bash
pip install placebos
```

Or install from source:

```bash
git clone https://github.com/Adam-Rohde/PlaceboLMPythonn.git
cd PlaceboLMPythonn
pip install -e .
```

## Quick Start

### Basic Usage

```python
from placebos import PlaceboLM

# Create a simple placebo
placebo = PlaceboLM(default_response="Hello, I am a placebo!")

# Generate a response
response = placebo.generate("What is the meaning of life?")
print(response.content)  # "Hello, I am a placebo!"
```

### Using Response Maps

```python
from placebos import PlaceboLM

# Map specific prompts to specific responses
placebo = PlaceboLM(
    response_map={
        "Hello": "Hi there!",
        "How are you?": "I'm doing great, thanks!",
    },
    default_response="I don't understand."
)

print(placebo.generate("Hello").content)  # "Hi there!"
print(placebo.generate("Random").content)  # "I don't understand."
```

### Custom Response Functions

```python
from placebos import PlaceboLM

# Use a function to generate dynamic responses
def echo_response(prompt):
    return f"You said: {prompt}"

placebo = PlaceboLM(response_func=echo_response)
print(placebo.generate("Hello world").content)  # "You said: Hello world"
```

### Chat-style Interface

```python
from placebos import PlaceboLM

placebo = PlaceboLM(default_response="I understand.")

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "How are you?"},
]

response = placebo.chat(messages)
print(response.content)  # "I understand."
```

## Provider-Specific Placebos

### OpenAI Format

```python
from placebos import OpenAIPlacebo

placebo = OpenAIPlacebo(
    model="gpt-4",
    default_response="This is a mock response."
)

# Returns OpenAI-compatible response format
response = placebo.create_chat_completion([
    {"role": "user", "content": "Hello"}
])

print(response["choices"][0]["message"]["content"])
```

### Anthropic Format

```python
from placebos import AnthropicPlacebo

placebo = AnthropicPlacebo(
    model="claude-3-opus-20240229",
    default_response="This is a mock response."
)

# Returns Anthropic-compatible response format
response = placebo.create_message([
    {"role": "user", "content": "Hello"}
])

print(response["content"][0]["text"])
```

## Testing Utilities

The `PlaceboLM` class includes built-in assertion methods for testing:

```python
from placebos import PlaceboLM

placebo = PlaceboLM(default_response="Test response")

# Use in your tests
placebo.generate("Test prompt")

# Assert the placebo was called
placebo.assert_called()

# Assert it was called with a specific prompt
placebo.assert_called_with("Test prompt")

# Assert call count
placebo.assert_call_count(1)

# Access call history
print(placebo.call_history)  # List of all calls with details
```

## Configuration

Use `PlaceboConfig` for advanced configuration:

```python
from placebos import PlaceboLM, PlaceboConfig

config = PlaceboConfig(
    default_response="Configured response",
    latency_ms=100,           # Simulate 100ms latency
    latency_variance_ms=20,   # +/- 20ms variance
    token_count=50,           # Report 50 tokens used
    random_seed=42,           # For reproducible behavior
)

placebo = PlaceboLM(config=config)
```

## License

MIT License
