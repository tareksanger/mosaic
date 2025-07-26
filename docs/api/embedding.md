# Embedding Classes

The embedding classes provide a unified interface for generating vector embeddings from text across different providers.

## BaseEmbedding

The abstract base class that all embedding implementations inherit from.

### Abstract Methods

- `embed(texts: List[str]) -> List[np.ndarray]`: Generate embeddings for a list of texts

## OpenAIEmbedding

The `OpenAIEmbedding` class provides integration with OpenAI's embedding models.

### Constructor

```python
from mosaic.core.ai.embedding import OpenAIEmbedding

embedding = OpenAIEmbedding(
    api_key: str,
    model: str = "text-embedding-ada-002",
    base_url: Optional[str] = None,
    timeout: int = 60
)
```

### Parameters

- `api_key` (str): Your OpenAI API key
- `model` (str): The embedding model to use (default: "text-embedding-ada-002")
- `base_url` (Optional[str]): Custom base URL for API calls
- `timeout` (int): Request timeout in seconds

### Methods

#### `embed(texts: List[str]) -> List[np.ndarray]`

Generates embeddings for a list of texts.

**Parameters:**
- `texts` (List[str]): List of texts to generate embeddings for

**Returns:**
- `List[np.ndarray]`: List of embedding vectors

**Example:**
```python
from mosaic.core.ai.embedding import OpenAIEmbedding
import numpy as np

embedding = OpenAIEmbedding(api_key="your-api-key")
texts = ["Hello world", "Goodbye world", "Python programming"]
embeddings = embedding.embed(texts)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {embeddings[0].shape}")
# Output: Number of embeddings: 3
# Output: Embedding dimension: (1536,)
```

### Supported Models

- `text-embedding-ada-002` (default, 1536 dimensions)
- `text-embedding-3-small` (1536 dimensions)
- `text-embedding-3-large` (3072 dimensions)

## SentenceTransformerEmbedding

The `SentenceTransformerEmbedding` class provides integration with Hugging Face's sentence-transformers library.

### Constructor

```python
from mosaic.core.ai.embedding import SentenceTransformerEmbedding

embedding = SentenceTransformerEmbedding(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    cache_folder: Optional[str] = None
)
```

### Parameters

- `model_name` (str): The sentence transformer model to use
- `device` (Optional[str]): Device to run on ("cpu", "cuda", etc.)
- `cache_folder` (Optional[str]): Folder to cache downloaded models

### Methods

#### `embed(texts: List[str]) -> List[np.ndarray]`

Generates embeddings for a list of texts.

**Example:**
```python
from mosaic.core.ai.embedding import SentenceTransformerEmbedding

embedding = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
texts = ["Hello world", "Goodbye world"]
embeddings = embedding.embed(texts)

print(f"Embedding dimension: {embeddings[0].shape}")
# Output: Embedding dimension: (384,)
```

### Popular Models

- `all-MiniLM-L6-v2` (384 dimensions, fast)
- `all-mpnet-base-v2` (768 dimensions, good quality)
- `all-MiniLM-L12-v2` (384 dimensions, better quality)
- `multi-qa-MiniLM-L6-cos-v1` (384 dimensions, optimized for QA)

## Embedding Factory

The `EmbeddingFactory` provides a convenient way to create embedding instances.

### Usage

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

## Advanced Usage

### Batch Processing

```python
from mosaic.core.ai.embedding import OpenAIEmbedding

embedding = OpenAIEmbedding(api_key="your-key")

# Process large batches
large_text_list = ["Text " + str(i) for i in range(1000)]
embeddings = embedding.embed(large_text_list)

print(f"Generated {len(embeddings)} embeddings")
```

### Embedding Similarity

```python
from mosaic.core.ai.embedding import OpenAIEmbedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding = OpenAIEmbedding(api_key="your-key")

texts = [
    "The cat is on the mat",
    "A cat sits on a mat",
    "The weather is sunny today"
]

embeddings = embedding.embed(texts)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)

print("Similarity matrix:")
print(similarity_matrix)
```

### Custom Model Configuration

```python
from mosaic.core.ai.embedding import SentenceTransformerEmbedding

# Use GPU if available
embedding = SentenceTransformerEmbedding(
    model_name="all-mpnet-base-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Custom cache location
embedding = SentenceTransformerEmbedding(
    model_name="all-MiniLM-L6-v2",
    cache_folder="/path/to/custom/cache"
)
```

### Error Handling

```python
from mosaic.core.ai.embedding import OpenAIEmbedding
from tenacity import RetryError

try:
    embedding = OpenAIEmbedding(api_key="invalid-key")
    embeddings = embedding.embed(["Hello world"])
except RetryError as e:
    print(f"Failed to generate embeddings: {e}")
```

## Performance Considerations

### OpenAI Embeddings
- ✅ Fast and reliable
- ✅ Consistent quality
- ❌ Requires API calls (cost and latency)
- ❌ Requires internet connection

### Sentence Transformers
- ✅ Runs locally (no API calls)
- ✅ Free to use
- ✅ Can run offline
- ❌ Requires more memory
- ❌ Slower for large batches
- ❌ Model download required

### Choosing the Right Model

- **Production with budget**: OpenAI embeddings
- **Development/testing**: Sentence transformers
- **Offline applications**: Sentence transformers
- **High throughput**: OpenAI embeddings
- **Privacy-sensitive**: Sentence transformers 