# Basic Usage Examples

## Simple Text Generation

### OpenAI GPT

```python
from mosaic.core.ai.llm import OpenAILLM

# Initialize LLM
llm = OpenAILLM(api_key="your-openai-api-key")

# Generate response
response = llm.generate("What is the capital of France?")
print(response.content)
# Output: The capital of France is Paris.
```

### Google Gemini

```python
from mosaic.core.ai.llm import GeminiLLM

# Initialize LLM
llm = GeminiLLM(api_key="your-google-api-key")

# Generate response
response = llm.generate("Explain machine learning in simple terms")
print(response.content)
```

## Using Embeddings

### OpenAI Embeddings

```python
from mosaic.core.ai.embedding import OpenAIEmbedding

# Initialize embedding model
embedding = OpenAIEmbedding(api_key="your-openai-api-key")

# Generate embeddings
texts = ["Hello world", "Goodbye world", "Python programming"]
embeddings = embedding.embed(texts)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {embeddings[0].shape}")
# Output: Number of embeddings: 3
# Output: Embedding dimension: (1536,)
```

### Sentence Transformers

```python
from mosaic.core.ai.embedding import SentenceTransformerEmbedding

# Initialize embedding model
embedding = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")

# Generate embeddings
texts = ["Hello world", "Goodbye world"]
embeddings = embedding.embed(texts)

print(f"Embedding dimension: {embeddings[0].shape}")
# Output: Embedding dimension: (384,)
```

## Token Counting

```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(api_key="your-api-key")
response = llm.generate("Count the tokens in this text")

print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

## Error Handling

```python
from mosaic.core.ai.llm import OpenAILLM
from tenacity import RetryError

try:
    llm = OpenAILLM(api_key="invalid-key")
    response = llm.generate("Hello")
except RetryError:
    print("Failed to connect to OpenAI API")
```

## Configuration

### Environment Variables

```python
import os
from mosaic.core.ai.llm import OpenAILLM

# Set API key via environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize without explicit API key
llm = OpenAILLM()  # Will use environment variable
response = llm.generate("Hello!")
```

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

## Batch Processing

### Multiple Prompts

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

for i, response in enumerate(responses):
    print(f"Response {i+1}: {response[:100]}...")
```

### Multiple Texts for Embeddings

```python
from mosaic.core.ai.embedding import OpenAIEmbedding

embedding = OpenAIEmbedding(api_key="your-key")

# Large batch of texts
texts = [f"Document {i}" for i in range(100)]
embeddings = embedding.embed(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {embeddings[0].shape[0]} dimensions")
```

## Using the Factory Pattern

### LLM Factory

```python
from mosaic.core.ai.llm import LLMFactory

# Create OpenAI LLM
openai_llm = LLMFactory.create(
    provider="openai",
    api_key="your-openai-key"
)

# Create Gemini LLM
gemini_llm = LLMFactory.create(
    provider="gemini",
    api_key="your-google-key"
)
```

### Embedding Factory

```python
from mosaic.core.ai.embedding import EmbeddingFactory

# Create OpenAI embedding
openai_embedding = EmbeddingFactory.create(
    provider="openai",
    api_key="your-openai-key"
)

# Create sentence transformer embedding
st_embedding = EmbeddingFactory.create(
    provider="sentence-transformers",
    model_name="all-MiniLM-L6-v2"
)
```

## Working with Responses

### Accessing Response Data

```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(api_key="your-key")
response = llm.generate("Tell me a joke")

# Access the generated text
print(f"Generated text: {response.content}")

# Access token usage
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
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

## Common Patterns

### Retry Logic

The library includes built-in retry logic for API calls:

```python
from mosaic.core.ai.llm import OpenAILLM

# Retry logic is automatically handled
llm = OpenAILLM(api_key="your-key")

# If the API call fails, it will retry automatically
response = llm.generate("Hello world")
```

### Logging

```python
import logging
from mosaic.core.ai.llm import OpenAILLM

# Set up logging
logging.basicConfig(level=logging.INFO)

llm = OpenAILLM(api_key="your-key")
response = llm.generate("Hello world")

# You'll see logs about API calls, retries, etc.
``` 