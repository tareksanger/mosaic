# Mosaic

A Python library for AI/ML operations with LLM and embedding support.

## Quick Start

```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(api_key="your-api-key")
response = llm.generate("Hello, world!")
print(response.content)
```

## Installation

```bash
pip install mosaic-mind
```

## What is Mosaic?
Mosaic is a Python library designed for building sophisticated multi-agent orchestration systems. It provides the foundation for creating intelligent, collaborative agents that can work together to solve complex tasks.

### Core Capabilities

- **Multi-Agent Orchestration**: Coordinate multiple specialized agents to work together
- **Intelligent Task Distribution**: Automatically assign and route tasks to appropriate agents
- **Agent Communication**: Enable agents to communicate, share information, and collaborate
- **Workflow Automation**: Define and execute complex multi-step workflows
- **AI Integration**: Seamless integration with various LLM providers (OpenAI, Google Gemini)
- **Embedding Support**: Vector-based knowledge representation and retrieval

### Key Features

- **Modular Agent Architecture**: Create specialized agents for different tasks
- **Dynamic Task Routing**: Intelligent task assignment based on agent capabilities
- **State Management**: Track and manage agent states and conversation history
- **Error Recovery**: Built-in retry logic and error handling for robust operation
- **Extensible Design**: Easy to add new agent types and orchestration patterns

## Documentation

- [📚 Full Documentation](docs/) - Complete API reference and guides
- [🚀 Getting Started](docs/getting-started.md) - Installation and basic usage
- [🔧 Contributing](CONTRIBUTING.md) - Development setup and guidelines

## Quick Examples

### Text Generation
```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(api_key="your-api-key")
response = llm.generate("Explain quantum computing in simple terms")
print(response.content)
```

### Embeddings
```python
from mosaic.core.ai.embedding import OpenAIEmbedding

embedding = OpenAIEmbedding(api_key="your-api-key")
embeddings = embedding.embed(["Hello world", "Goodbye world"])
print(embeddings)
```

### Token Counting
```python
from mosaic.core.ai.llm import OpenAILLM

llm = OpenAILLM(api_key="your-api-key")
response = await llm.response("Count the tokens in this text")
print(f"Tokens used: {response.usage.total_tokens}")
```

## License

MIT License - see LICENSE file for details.MIT License - see LICENSE file for details.
