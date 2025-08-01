# Mosaic Agents

The Mosaic Agents module provides a sophisticated framework for creating and orchestrating AI agents with specific capabilities and organizational patterns. The module consists of two main components: **Base Agents** and **Mosaic Agents**.

## Overview

### Base Agents
Base agents are intelligent agents created dynamically based on user objectives and available functions. The system uses an LLM to automatically generate appropriate agents with specific instructions and capabilities based on the provided objective and function list.

### Mosaic Agents
Mosaic agents extend base agents with organizational patterns and coordination capabilities. Currently, the system supports **handoffs** as the primary coordination pattern, allowing agents to seamlessly transfer control and context between each other.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │   Base Agents   │    │  Mosaic Agents  │
│                 │    │                 │    │                 │
│ • Objective     │───▶│ • Instructions  │───▶│ • Handoffs      │
│ • Functions     │    │ • Capabilities  │    │ • Coordination  │
│ • Requirements  │    │ • Roles         │    │ • Patterns      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Base Agents

### Purpose
Base agents are dynamically created agents that receive:
- **Objective**: The main goal or task to accomplish
- **Functions**: Available capabilities/tools the agent can use
- **Context**: Additional information and requirements

The LLM then creates appropriate agents with:
- Specific instructions tailored to the objective
- Assigned functions from the provided list
- Clear roles and responsibilities

### Key Components

#### BaseAgent
```python
class BaseAgent(BaseModel):
    name: str                    # Agent identifier
    instructions: str            # Specific instructions for the agent
    functions: List[str]         # Assigned functions from available list
    reason: str                  # Justification for agent creation and function assignment
```

#### ManagerAgent
```python
class ManagerAgent(BaseModel):
    name: str                    # Manager agent identifier
    instructions: str            # Management and coordination instructions
    functions: List[str]         # Planning and final answer generation functions
```

#### EvaluatorAgent
```python
class EvaluatorAgent(BaseModel):
    name: str                    # Evaluator agent identifier
    instructions: str            # Evaluation instructions
    evaluation_criteria: List[EvaluationResults]  # Criteria for assessment
```

#### BaseAgents (Complete System)
```python
class BaseAgents(BaseModel):
    agents: List[BaseAgent]      # Worker agents
    manager_agent: ManagerAgent  # Coordination agent
    evaluator_agent: EvaluatorAgent  # Quality assessment agent
    plan: Plan                   # Execution plan with steps
```

### Usage Example

```python
from mosaic.core.ai.agents.base_agent import BaseAgents

# Define your objective and available functions
objective = "Analyze customer feedback and generate improvement recommendations"
available_functions = ["analyze_sentiment", "extract_keywords", "generate_report"]

# LLM creates appropriate agents
base_agents = BaseAgents(
    agents=[
        BaseAgent(
            name="SentimentAnalyzer",
            instructions="Analyze sentiment of customer feedback",
            functions=["analyze_sentiment"],
            reason="Sentiment analysis is crucial for understanding customer satisfaction"
        ),
        BaseAgent(
            name="KeywordExtractor", 
            instructions="Extract key themes from feedback",
            functions=["extract_keywords"],
            reason="Keyword extraction helps identify common issues and trends"
        )
    ],
    manager_agent=ManagerAgent(
        name="FeedbackManager",
        instructions="Coordinate analysis and generate final recommendations",
        functions=["plan_for_next_steps", "generate_final_answer"]
    ),
    # ... evaluator and plan
)
```

## Mosaic Agents

### Purpose
Mosaic agents organize base agents into coordinated patterns, currently supporting **handoffs** as the primary coordination mechanism. This allows for:

- **Seamless transitions** between agents
- **Context preservation** during handoffs
- **Flexible workflow** execution
- **Scalable multi-agent** systems

### Key Features

#### Handoff Support
- Worker agents can hand off to manager agents
- Manager agents can hand off to any worker agent
- Automatic handoff instruction injection
- Context preservation during transitions

#### Agent Coordination
- Centralized management through manager agents
- Plan-driven execution
- Function tool integration
- Async and sync execution support

### Usage Example

```python
from mosaic.core.ai.agents.mosaic_agent import MosaicAgent

# Create function mapping
functions_dict = {
    "analyze_sentiment": analyze_sentiment_function,
    "extract_keywords": extract_keywords_function,
    "generate_report": generate_report_function,
    "plan_for_next_steps": plan_function,
    "generate_final_answer": final_answer_function
}

# Create mosaic agent from base agents
mosaic_agent = MosaicAgent(base_agents, functions_dict)

# Run the system
result = mosaic_agent.run("Analyze this customer feedback: [feedback text]")

# Async execution
result = await mosaic_agent.arun("Analyze this customer feedback: [feedback text]")
```

## Agent Creation Process

### 1. Objective Analysis
The LLM analyzes the user's objective and available functions to understand:
- What needs to be accomplished
- What capabilities are available
- How to best organize the work

### 2. Agent Generation
Based on the analysis, the LLM creates:
- **Worker agents** with specific roles and assigned functions
- **Manager agent** for coordination and planning
- **Evaluator agent** for quality assessment
- **Execution plan** with clear steps

### 3. Pattern Application
The mosaic agent system applies organizational patterns:
- **Handoffs**: Enables seamless agent transitions
- **Function mapping**: Connects agent functions to actual implementations
- **Coordination**: Manages agent interactions and workflow

## Supported Patterns

### Handoffs (Currently Supported)
- **Worker → Manager**: Task completion handoffs
- **Manager → Worker**: Task assignment handoffs
- **Context Preservation**: Maintains conversation state
- **Automatic Routing**: Smart handoff decisions

### Future Patterns (Planned)
- **Parallel Execution**: Multiple agents working simultaneously
- **Hierarchical Coordination**: Multi-level agent management
- **Dynamic Routing**: Adaptive agent selection
- **Load Balancing**: Distributed workload management

## Best Practices

### Agent Design
1. **Single Responsibility**: Each agent should have a clear, focused role
2. **Function Assignment**: Assign only necessary functions to each agent
3. **Clear Instructions**: Provide specific, actionable instructions
4. **Reason Documentation**: Explain why each agent and function assignment makes sense

### System Configuration
1. **Function Mapping**: Ensure all referenced functions exist in the functions_dict
2. **Plan Clarity**: Create clear, step-by-step execution plans
3. **Evaluation Criteria**: Define specific quality assessment metrics
4. **Error Handling**: Implement proper error handling and recovery

### Performance Optimization
1. **Async Execution**: Use async methods for better performance
2. **Tracing**: Enable tracing for debugging and monitoring
3. **Resource Management**: Monitor agent resource usage
4. **Caching**: Implement caching for repeated operations

## Error Handling

The system includes comprehensive error handling:
- **Function Validation**: Ensures all referenced functions exist
- **Agent Validation**: Validates agent structure and relationships
- **Execution Monitoring**: Tracks agent execution and handoffs
- **Graceful Degradation**: Handles failures without system collapse

## Monitoring and Debugging

### Tracing Support
```python
# Enable tracing for debugging
result = mosaic_agent.run(prompt, trace_name="customer_feedback_analysis")
```

### Agent Inspection
```python
# List all agents
agent_names = mosaic_agent.list_agents()

# Get specific agent
agent = mosaic_agent.get_agent_by_name("SentimentAnalyzer")
```

## Integration

The agents module integrates with:
- **OpenAI Agents**: Primary agent implementation
- **Function Tools**: Capability integration
- **Pydantic Models**: Data validation and structure
- **Async Support**: Non-blocking execution
- **Tracing**: Debugging and monitoring

## Future Enhancements

- **Additional Patterns**: Parallel execution, hierarchical coordination
- **Advanced Routing**: Dynamic agent selection based on workload
- **Performance Optimization**: Caching, batching, and optimization
- **Extended Monitoring**: Advanced analytics and insights
- **Pattern Templates**: Pre-built coordination patterns 