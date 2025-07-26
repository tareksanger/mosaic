# Best Practices

## API Key Management

### Environment Variables

Always use environment variables for API keys:

```python
import os
from mosaic.core.ai.llm import OpenAILLM

# Good: Use environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
llm = OpenAILLM()  # Will use environment variable

# Bad: Hardcode API keys
llm = OpenAILLM(api_key="sk-1234567890abcdef")  # Don't do this!
```

### Configuration Files

Use `.env` files for local development:

```env
# .env
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
```

```python
from dotenv import load_dotenv
load_dotenv()

from mosaic.core.ai.llm import OpenAILLM
llm = OpenAILLM()  # Uses environment variable
```

## Error Handling

### Graceful Degradation

```python
from mosaic.core.ai.llm import OpenAILLM
from tenacity import RetryError

def generate_response(prompt: str, fallback: str = "I'm sorry, I can't help right now."):
    try:
        llm = OpenAILLM()
        response = llm.generate(prompt)
        return response.content
    except RetryError:
        return fallback
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return fallback
```


## Performance Optimization

### Batch Processing

For multiple requests, batch them when possible:

```python
from mosaic.core.ai.embedding import OpenAIEmbedding

# Good: Batch embedding requests
embedding = OpenAIEmbedding()
texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
embeddings = embedding.embed(texts)  # Single API call

# Bad: Multiple individual requests
for text in texts:
    embedding_result = embedding.embed([text])  # Multiple API calls
```

### Caching

Implement caching for expensive operations:

```python
import functools
from mosaic.core.ai.embedding import OpenAIEmbedding

@functools.lru_cache(maxsize=1000)
def get_embedding(text: str):
    embedding = OpenAIEmbedding()
    return embedding.embed([text])[0]

# Reuse embeddings for repeated texts
embedding1 = get_embedding("Hello world")
embedding2 = get_embedding("Hello world")  # Cached result
```

## Token Management

### Monitor Usage

```python
from mosaic.core.ai.llm import OpenAILLM

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
    
    def track_usage(self, response):
        self.total_tokens += response.usage.total_tokens
        print(f"Total tokens used: {self.total_tokens}")

tracker = TokenTracker()
llm = OpenAILLM(api_key="your-key")

response = llm.generate("Hello")
tracker.track_usage(response)
```