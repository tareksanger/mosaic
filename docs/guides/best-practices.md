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
        llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"))
        response = llm.generate(prompt)
        return response.content
    except RetryError:
        return fallback
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return fallback
```

### Retry Configuration

```python
from mosaic.core.ai.llm import OpenAILLM

# Custom retry configuration
llm = OpenAILLM(
    api_key="your-key",
    timeout=120,  # Longer timeout for complex requests
)
```

## Performance Optimization

### Batch Processing

For multiple requests, batch them when possible:

```python
from mosaic.core.ai.embedding import OpenAIEmbedding

# Good: Batch embedding requests
embedding = OpenAIEmbedding(api_key="your-key")
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
    embedding = OpenAIEmbedding(api_key="your-key")
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

### Cost Optimization

```python
from mosaic.core.ai.llm import OpenAILLM

# Use appropriate models for different tasks
def get_llm_for_task(task_type: str):
    if task_type == "simple":
        return OpenAILLM(model="gpt-3.5-turbo")  # Cheaper
    elif task_type == "complex":
        return OpenAILLM(model="gpt-4")  # More capable
    else:
        return OpenAILLM(model="gpt-3.5-turbo")  # Default
```

## Production Considerations

### Logging

```python
import logging
from mosaic.core.ai.llm import OpenAILLM

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def generate_with_logging(prompt: str):
    logger.info(f"Generating response for prompt: {prompt[:50]}...")
    
    try:
        llm = OpenAILLM(api_key="your-key")
        response = llm.generate(prompt)
        
        logger.info(f"Generated response with {response.usage.total_tokens} tokens")
        return response.content
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        raise
```

### Health Checks

```python
from mosaic.core.ai.llm import OpenAILLM

def health_check():
    try:
        llm = OpenAILLM(api_key="your-key")
        response = llm.generate("test")
        return True
    except Exception:
        return False

# Use in your application
if not health_check():
    print("LLM service is not available")
```

## Security Best Practices

### Input Validation

```python
def sanitize_prompt(prompt: str) -> str:
    # Remove potentially harmful content
    if len(prompt) > 10000:  # Limit prompt length
        raise ValueError("Prompt too long")
    
    # Remove sensitive information patterns
    import re
    prompt = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD_NUMBER]', prompt)
    
    return prompt

# Use in your application
llm = OpenAILLM(api_key="your-key")
safe_prompt = sanitize_prompt(user_input)
response = llm.generate(safe_prompt)
```

### Rate Limiting

```python
import time
from functools import wraps

def rate_limit(calls_per_minute: int = 60):
    def decorator(func):
        last_call = 0
        min_interval = 60 / calls_per_minute
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_call
            current_time = time.time()
            
            if current_time - last_call < min_interval:
                time.sleep(min_interval - (current_time - last_call))
            
            last_call = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_minute=30)
def generate_response(prompt: str):
    llm = OpenAILLM(api_key="your-key")
    return llm.generate(prompt)
```

## Testing

### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch
from mosaic.core.ai.llm import OpenAILLM

def test_llm_generation():
    with patch('mosaic.core.ai.llm.openai.OpenAI') as mock_openai:
        # Mock the response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        llm = OpenAILLM(api_key="test-key")
        response = llm.generate("Test prompt")
        
        assert response.content == "Test response"
        assert response.usage.total_tokens == 15
```

### Integration Tests

```python
import pytest
from mosaic.core.ai.embedding import SentenceTransformerEmbedding

def test_embedding_generation():
    # Use sentence transformers for testing (no API key needed)
    embedding = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
    
    texts = ["Hello world", "Goodbye world"]
    embeddings = embedding.embed(texts)
    
    assert len(embeddings) == 2
    assert embeddings[0].shape[0] == 384  # MiniLM dimension
```

## Monitoring and Observability

### Metrics Collection

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class LLMMetrics:
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    average_response_time: float = 0.0

class LLMMonitor:
    def __init__(self):
        self.metrics = LLMMetrics()
        self.response_times: List[float] = []
    
    def record_request(self, response, response_time: float):
        self.metrics.total_requests += 1
        self.metrics.total_tokens += response.usage.total_tokens
        self.response_times.append(response_time)
        self.metrics.average_response_time = sum(self.response_times) / len(self.response_times)
    
    def record_error(self):
        self.metrics.total_errors += 1
    
    def get_metrics(self):
        return self.metrics

# Usage
monitor = LLMMonitor()
llm = OpenAILLM(api_key="your-key")

start_time = time.time()
try:
    response = llm.generate("Hello")
    response_time = time.time() - start_time
    monitor.record_request(response, response_time)
except Exception:
    monitor.record_error()

print(f"Metrics: {monitor.get_metrics()}")
``` 