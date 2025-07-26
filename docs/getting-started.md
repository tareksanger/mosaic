# Getting Started

## Installation

### From PyPI

```bash
pip install mosaic-mind
```

### With Optional Dependencies

For sentence transformer embeddings:

```bash
pip install mosaic-mind[sentence-transformers]
```

For all optional dependencies:

```bash
pip install mosaic-mind[all]
```

## Quick Start

### 1. Set Up API Keys

You'll need API keys for the services you want to use:

```python
import os

# OpenAI
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Google (for Gemini)
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
```

### 2. Basic Text Generation

```python
from mosaic.core.ai.llm import OpenAILLM

# Initialize the LLM
llm = OpenAILLM(api_key="your-openai-api-key")

# Generate a response
response = llm.generate("What is the capital of France?")
print(response.content)
# Output: The capital of France is Paris.
```

### 3. Using Embeddings

```python
from mosaic.core.ai.embedding import OpenAIEmbedding

# Initialize the embedding model
embedding = OpenAIEmbedding(api_key="your-openai-api-key")

# Generate embeddings
texts = ["Hello world", "Goodbye world"]
embeddings = embedding.embed(texts)

print(f"Embedding shape: {embeddings[0].shape}")
# Output: Embedding shape: (1536,)
```

### 4. Token Counting

```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(api_key="your-openai-api-key")
response = llm.generate("Count the tokens in this text")

print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

## Configuration

### Environment Variables

You can set API keys via environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

### Configuration Files

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
```

Then load it in your code:

```python
from dotenv import load_dotenv
load_dotenv()

from mosaic.core.ai.llm import OpenAILLM
llm = OpenAILLM()  # Will use environment variable
```

## Next Steps

- Check out the [API Reference](api/) for detailed class documentation
- Explore [Examples](examples/) for more complex use cases
- Read the [Guides](guides/) for best practices and advanced topics 