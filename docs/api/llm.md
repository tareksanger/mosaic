# LLM Classes

The LLM (Large Language Model) classes provide a unified interface for text generation across different providers.

## BaseLLM

The abstract base class that all LLM implementations inherit from.

### Abstract Methods

- `generate(prompt: str) -> Response`: Generate a response for the given prompt

## OpenAILLM

The `OpenAILLM` class provides integration with OpenAI's GPT models.

### Constructor

```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(
    api_key: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    base_url: Optional[str] = None,
    timeout: int = 60
)
```

### Parameters

- `api_key` (str): Your OpenAI API key
- `model` (str): The model to use (default: "gpt-3.5-turbo")
- `temperature` (float): Controls randomness (0.0 = deterministic, 1.0 = very random)
- `max_tokens` (int): Maximum number of tokens to generate
- `base_url` (Optional[str]): Custom base URL for API calls
- `timeout` (int): Request timeout in seconds

### Methods

#### `generate(prompt: str) -> Response`

Generates a response for the given prompt.

**Parameters:**
- `prompt` (str): The input text to generate a response for

**Returns:**
- `Response`: Object containing the generated text and metadata

**Example:**
```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(api_key="your-api-key")
response = llm.generate("Explain quantum computing")
print(response.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Response Object

The `Response` object contains:

- `content` (str): The generated text
- `usage` (Usage): Token usage information
  - `prompt_tokens` (int): Tokens in the input
  - `completion_tokens` (int): Tokens in the output
  - `total_tokens` (int): Total tokens used

### Error Handling

```python
from mosaic.core.ai.llm import OpenAILLM
from tenacity import RetryError

try:
    llm = OpenAILLM(api_key="invalid-key")
    response = llm.generate("Hello")
except RetryError as e:
    print(f"Failed to connect: {e}")
```

## GeminiLLM

The `GeminiLLM` class provides integration with Google's Gemini models.

### Constructor

```python
from mosaic.core.ai.llm import GeminiLLM

llm = GeminiLLM(
    api_key: str,
    model: str = "gemini-pro",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    timeout: int = 60
)
```

### Parameters

- `api_key` (str): Your Google API key
- `model` (str): The model to use (default: "gemini-pro")
- `temperature` (float): Controls randomness (0.0 = deterministic, 1.0 = very random)
- `max_tokens` (int): Maximum number of tokens to generate
- `timeout` (int): Request timeout in seconds

### Methods

#### `generate(prompt: str) -> Response`

Generates a response for the given prompt.

**Example:**
```python
from mosaic.core.ai.llm import GeminiLLM

llm = GeminiLLM(api_key="your-google-api-key")
response = llm.generate("What is machine learning?")
print(response.content)
```

### Supported Models

#### OpenAI Models
- `gpt-3.5-turbo`
- `gpt-4`
- `gpt-4-turbo`
- `gpt-4o`

#### Gemini Models
- `gemini-pro`
- `gemini-pro-vision`

## Advanced Usage

### Custom Configuration

```python
from mosaic.core.ai.llm import OpenAILLM

# Custom configuration
llm = OpenAILLM(
    api_key="your-key",
    model="gpt-4",
    temperature=0.1,  # More deterministic
    max_tokens=500,
    timeout=120
)
```

### Batch Processing

```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(api_key="your-key")
prompts = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]

responses = []
for prompt in prompts:
    response = llm.generate(prompt)
    responses.append(response.content)
```

### Token Usage Tracking

```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(api_key="your-key")
total_tokens = 0

for i in range(5):
    response = llm.generate(f"Generate response {i}")
    total_tokens += response.usage.total_tokens
    print(f"Response {i}: {response.usage.total_tokens} tokens")

print(f"Total tokens used: {total_tokens}")
``` 