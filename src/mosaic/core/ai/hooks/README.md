# Mosaic Hooks

The Mosaic Hooks module provides comprehensive error handling and validation capabilities for multi-agent workflows. The hooks system monitors agent execution, validates operations, and provides detailed error tracking and reporting.

## Overview

The hooks system is designed to catch and handle various error conditions that can occur during agent runs, including:
- **Validation errors** during agent execution
- **Tool usage violations** (missing required tools)
- **Handoff validation** (invalid agent transitions)
- **Workflow state violations** (business logic errors)
- **Execution monitoring** and debugging

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agent Run     │    │   Error Hooks   │    ┌   Validation    │
│                 │    │                 │    │                 │
│ • Agent Start   │───▶│ • Event Tracking│───▶│ • Tool Usage    │
│ • Tool Calls    │    │ • Error Capture │    │ • Handoffs      │
│ • Handoffs      │    │ • State Monitor │    │ • Workflow      │
│ • Agent End     │    │ • Validation    │    │ • Business Logic│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Error Types

### AgentValidationError (Base Class)
The foundation for all agent validation errors with context tracking.

```python
class AgentValidationError(Exception):
    def __init__(self, agent_name, error_message, context=None):
        self.agent_name = agent_name
        self.error_message = error_message
        self.context = context or {}
```

### Specific Error Types

#### RequiredToolNotCalledError
Raised when an agent fails to call required tools before proceeding.

```python
class RequiredToolNotCalledError(AgentValidationError):
    def __init__(self, agent_name, missing_tools, context=None):
        self.missing_tools = missing_tools if isinstance(missing_tools, list) else [missing_tools]
```

**Example**: Agent "Buyer Agent" must call `search_topic` before handoff.

#### PrematureCompletionError
Raised when an agent completes without fulfilling required conditions.

```python
class PrematureCompletionError(AgentValidationError):
    def __init__(self, agent_name, missing_conditions, context=None):
        self.missing_conditions = missing_conditions if isinstance(missing_conditions, list) else [missing_conditions]
```

**Example**: Agent completes without generating required output.

#### InvalidHandoffError
Raised when an agent attempts an invalid handoff.

```python
class InvalidHandoffError(AgentValidationError):
    def __init__(self, agent_name, handoff_target, reason, context=None):
        self.handoff_target = handoff_target
        self.reason = reason
```

**Example**: Agent tries to hand off to unauthorized target agent.

#### AgentStateViolationError
Raised when an agent violates workflow state rules.

```python
class AgentStateViolationError(AgentValidationError):
    def __init__(self, agent_name, violated_rule, current_state=None, context=None):
        self.violated_rule = violated_rule
        self.current_state = current_state
```

**Example**: Agent violates business logic or state constraints.

#### WorkflowIncompleteError
Raised when workflow ends without reaching required completion state.

```python
class WorkflowIncompleteError(AgentValidationError):
    def __init__(self, agent_name, expected_completion, actual_state, context=None):
        self.expected_completion = expected_completion
        self.actual_state = actual_state
```

**Example**: Workflow ends without final agreement or conclusion.

#### MaxAttemptsExceededError
Raised when validation errors exceed maximum retry attempts.

```python
class MaxAttemptsExceededError(Exception):
    def __init__(self, max_attempts, error_history, last_error):
        self.max_attempts = max_attempts
        self.error_history = error_history
        self.last_error = last_error
```

## AgentErrorValidationHooks

The main hook class that provides comprehensive error handling and validation.

### Key Features

#### Unified Event Tracking
All events, states, and validations are tracked in a single comprehensive dictionary:

```python
self.workflow_events = {
    "agents": {},           # agent_name -> agent info
    "tools": [],           # list of tool events
    "handoffs": [],        # list of handoff events  
    "outputs": [],         # list of agent outputs
    "errors": [],          # list of validation errors
    "error_events": []     # list of detailed error events with context
}
```

#### Agent Requirements Configuration
Define agent-specific requirements and validation rules:

```python
self.requirements = {
    "Buyer Agent": {
        "required_tools": ["search_topic"], 
        "valid_targets": ["Negotiation Orchestrator"], 
        "min_tools": 1
    },
    "Seller Agent": {
        "required_tools": ["search_topic"], 
        "valid_targets": ["Negotiation Orchestrator"], 
        "min_tools": 1
    },
    "Negotiation Orchestrator": {
        "required_tools": [], 
        "valid_targets": ["Buyer Agent", "Seller Agent"], 
        "min_tools": 0
    }
}
```

### Hook Methods

#### on_agent_start
Initializes agent tracking and displays requirements.

```python
async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
    """Initialize agent in unified tracking"""
    self._ensure_agent_exists(agent.name)
    # Display agent requirements and start tracking
```

#### on_agent_end
Records agent completion and validates final state.

```python
async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
    """Record agent completion in unified tracking"""
    self._record_output_event(agent.name, output)
    # Validate completion requirements
```

#### on_tool_start
Records tool usage and validates tool requirements.

```python
async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
    """Record tool start in unified tracking"""
    self._record_tool_event(agent.name, tool.name, "start")
    # Check current state and requirements
```

#### on_tool_end
Records tool completion and updates agent readiness.

```python
async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str) -> None:
    """Record tool completion in unified tracking"""
    self._record_tool_event(agent.name, tool.name, "end", result)
    # Update agent readiness for handoff
```

#### on_handoff
Validates handoff requirements and records transitions.

```python
async def on_handoff(self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent) -> None:
    """Validate and record handoff using unified tracking"""
    # Validate agent readiness and handoff targets
    # Record successful handoff or raise validation error
```

### Validation Methods

#### _check_agent_ready
Validates if an agent meets all requirements for handoff.

```python
def _check_agent_ready(self, agent_name: str, raise_exceptions: bool = False) -> tuple[bool, list[str]]:
    """Check if agent meets all requirements for handoff"""
    # Validate required tools, minimum tools, and other requirements
    # Returns (is_ready, list_of_errors)
```

#### _check_valid_handoff
Validates if a handoff between agents is allowed.

```python
def _check_valid_handoff(self, from_agent: str, to_agent: str) -> tuple[bool, str]:
    """Check if handoff from one agent to another is valid"""
    # Validate handoff targets and business rules
    # Returns (is_valid, error_message)
```

## Usage Examples

### Basic Hook Integration

```python
from mosaic.core.ai.hooks.error_hooks import AgentErrorValidationHooks

# Create hooks instance
hooks = AgentErrorValidationHooks()

# Use with agent runner
runner = Runner()
result = await runner.run(agent, input_items, hooks=hooks)
```

### Custom Requirements Configuration

```python
# Customize agent requirements
hooks = AgentErrorValidationHooks()
hooks.requirements = {
    "Data Analyzer": {
        "required_tools": ["analyze_data", "generate_report"],
        "valid_targets": ["Report Manager"],
        "min_tools": 2
    },
    "Report Manager": {
        "required_tools": ["format_report"],
        "valid_targets": ["Data Analyzer"],
        "min_tools": 1
    }
}
```

### Error Handling and Recovery

```python
try:
    result = await runner.run(agent, input_items, hooks=hooks)
except RequiredToolNotCalledError as e:
    print(f"Agent {e.agent_name} failed to call required tools: {e.missing_tools}")
    # Implement retry logic or fallback
except InvalidHandoffError as e:
    print(f"Invalid handoff from {e.agent_name} to {e.handoff_target}: {e.reason}")
    # Implement handoff correction
except MaxAttemptsExceededError as e:
    print(f"Validation failed after {e.max_attempts} attempts")
    # Implement workflow termination or escalation
```

## Monitoring and Reporting

### Real-time Debugging

The hooks provide comprehensive real-time debugging output:

```
=================== AGENT START ===================
### 1: Agent Buyer Agent started
📋 Required tools: ['search_topic']
🔢 Minimum tools: 1

=================== TOOL START ===================
### 2: Tool search_topic started by Buyer Agent
🔧 Current tools: ['search_topic']
📊 Unique count: 1
✅ All requirements met!
```

### Final Summary

```python
# Get comprehensive workflow summary
summary = hooks.get_unified_summary()

# Save workflow log
hooks.save_workflow_log("workflow_analysis.json")

# Print final summary
hooks.print_final_summary()
```

### Error Analysis

```python
# Get detailed error events
error_events = hooks.get_error_events()

# Get error summary
error_summary = hooks.get_error_summary()

# Example output:
{
    "total_error_events": 2,
    "error_types": {
        "RequiredToolNotCalledError": 1,
        "InvalidHandoffError": 1
    },
    "agents_with_errors": {
        "Buyer Agent": 1,
        "Seller Agent": 1
    }
}
```

## Configuration Options

### Agent Requirements

Configure validation rules for each agent:

- **required_tools**: List of tools that must be called
- **valid_targets**: List of agents this agent can hand off to
- **min_tools**: Minimum number of unique tools to call
- **max_tools**: Maximum number of tool calls allowed
- **required_outputs**: Required output formats or content

### Validation Settings

- **raise_exceptions**: Whether to raise exceptions on validation failures
- **max_attempts**: Maximum retry attempts for failed validations
- **strict_mode**: Enforce all validation rules strictly
- **debug_mode**: Enable detailed debugging output

## Best Practices

### Error Handling
1. **Graceful Degradation**: Handle errors without workflow collapse
2. **Retry Logic**: Implement intelligent retry mechanisms
3. **Fallback Strategies**: Provide alternative execution paths
4. **Error Reporting**: Log detailed error information for analysis

### Validation Design
1. **Clear Requirements**: Define explicit agent requirements
2. **Business Logic**: Enforce workflow business rules
3. **State Management**: Track and validate workflow state
4. **Handoff Rules**: Define clear handoff permissions and conditions

### Monitoring
1. **Real-time Tracking**: Monitor agent execution in real-time
2. **Performance Metrics**: Track token usage and execution time
3. **Error Patterns**: Identify and analyze error patterns
4. **Workflow Analytics**: Generate comprehensive workflow reports

## Integration

The hooks system integrates with:
- **OpenAI Agents**: Primary agent implementation
- **Runner**: Agent execution engine
- **Tool System**: Function and tool validation
- **Handoff System**: Agent transition validation
- **Logging**: Comprehensive event logging and debugging 