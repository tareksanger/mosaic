from typing import Any
from pydantic import BaseModel
from agents import (
    Agent, Runner, RunContextWrapper, input_guardrail, output_guardrail,
    GuardrailFunctionOutput, TResponseInputItem, RunHooks,
    InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, handoff
)
from agents import Agent, RunContextWrapper, RunHooks, Runner, Tool, Usage, function_tool, ModelSettings
from agents import WebSearchTool
from agents import (
    Agent, Runner, RunContextWrapper, input_guardrail, output_guardrail,
    GuardrailFunctionOutput, TResponseInputItem, RunHooks,
    InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, handoff
)


class AgentValidationError(Exception):
    """Base class for all agent validation errors in multi-agent workflows"""
    def __init__(self, agent_name, error_message, context=None):
        self.agent_name = agent_name
        self.error_message = error_message
        self.context = context or {}
        super().__init__(f"Agent '{agent_name}': {error_message}")

class RequiredToolNotCalledError(AgentValidationError):
    """Agent failed to call required tool(s) before proceeding"""
    def __init__(self, agent_name, missing_tools, context=None):
        self.missing_tools = missing_tools if isinstance(missing_tools, list) else [missing_tools]
        super().__init__(
            agent_name, 
            f"Must call required tool(s): {', '.join(self.missing_tools)}", 
            context
        )

class PrematureCompletionError(AgentValidationError):
    """Agent completed execution without fulfilling required conditions"""
    def __init__(self, agent_name, missing_conditions, context=None):
        self.missing_conditions = missing_conditions if isinstance(missing_conditions, list) else [missing_conditions]
        super().__init__(
            agent_name,
            f"Completed without fulfilling: {', '.join(self.missing_conditions)}",
            context
        )

class InvalidHandoffError(AgentValidationError):
    """Agent attempted invalid handoff (wrong target, missing prerequisites, etc.)"""
    def __init__(self, agent_name, handoff_target, reason, context=None):
        self.handoff_target = handoff_target
        self.reason = reason
        super().__init__(
            agent_name,
            f"Invalid handoff to '{handoff_target}': {reason}",
            context
        )

class AgentStateViolationError(AgentValidationError):
    """Agent violated workflow state rules or business logic"""
    def __init__(self, agent_name, violated_rule, current_state=None, context=None):
        self.violated_rule = violated_rule
        self.current_state = current_state
        super().__init__(
            agent_name,
            f"State violation: {violated_rule}" + (f" (current state: {current_state})" if current_state else ""),
            context
        )

class WorkflowIncompleteError(AgentValidationError):
    """Workflow ended without reaching required completion state"""
    def __init__(self, agent_name, expected_completion, actual_state, context=None):
        self.expected_completion = expected_completion
        self.actual_state = actual_state
        super().__init__(
            agent_name,
            f"Workflow incomplete: expected '{expected_completion}', got '{actual_state}'",
            context
        )

class MaxAttemptsExceededError(Exception):
    """Validation errors exceeded maximum retry attempts"""
    def __init__(self, max_attempts, error_history, last_error):
        self.max_attempts = max_attempts
        self.error_history = error_history
        self.last_error = last_error
        super().__init__(
            f"Validation failed after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )


class AgentErrorValidationHooks(RunHooks):
    """
    Unified event tracking system using a single comprehensive dictionary.
    All events, states, and validations tracked in one place.
    """
    
    def __init__(self):
        # Event tracking
        self.event_counter = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # SINGLE UNIFIED TRACKING DICTIONARY
        self.workflow_events = {
            "agents": {},           # agent_name -> agent info
            "tools": [],           # list of tool events
            "handoffs": [],        # list of handoff events  
            "outputs": [],         # list of agent outputs
            "errors": [],          # list of validation errors
            "error_events": []     # list of detailed error events with context
        }
        
        # Simple agent requirements
        self.requirements = {
            "Buyer Agent": {"required_tools": ["search_topic"], "valid_targets": ["Negotiation Orchestrator"], "min_tools": 1},
            "Seller Agent": {"required_tools": ["search_topic"], "valid_targets": ["Negotiation Orchestrator"], "min_tools": 1},
            "Negotiation Orchestrator": {"required_tools": [], "valid_targets": ["Buyer Agent", "Seller Agent"], "min_tools": 0}
        }

    def _get_timestamp(self):
        import datetime
        return datetime.datetime.now().isoformat()

    def _safe_add_tokens(self, context: RunContextWrapper):
        if hasattr(context, 'usage') and context.usage:
            if hasattr(context.usage, 'input_tokens'):
                self.total_input_tokens += context.usage.input_tokens
            if hasattr(context.usage, 'output_tokens'):
                self.total_output_tokens += context.usage.output_tokens

    def _ensure_agent_exists(self, agent_name: str):
        """Initialize agent in tracking dict if not exists"""
        if agent_name not in self.workflow_events["agents"]:
            self.workflow_events["agents"][agent_name] = {
                "name": agent_name,
                "start_time": self._get_timestamp(),
                "tools_called": [],        # [(tool_name, timestamp, result), ...]
                "unique_tools": set(),     # {tool_name, ...}
                "tool_count": 0,
                "unique_tool_count": 0,
                "handoffs_from": [],       # [to_agent, ...]
                "handoffs_to": [],         # [from_agent, ...]
                "outputs": [],             # [output, ...]
                "ready_for_handoff": False,
                "requirements": self.requirements.get(agent_name, {}),
                "current_errors": []
            }

    def _record_tool_event(self, agent_name: str, tool_name: str, event_type: str, result: str = None):
        """Record tool event in unified dict"""
        self._ensure_agent_exists(agent_name)
        
        tool_event = {
            "event_id": self.event_counter,
            "timestamp": self._get_timestamp(),
            "agent_name": agent_name,
            "tool_name": tool_name,
            "event_type": event_type,  # "start" or "end"
            "result": result
        }
        
        # Add to global tools list
        self.workflow_events["tools"].append(tool_event)
        
        # Update agent-specific tracking
        agent = self.workflow_events["agents"][agent_name]
        
        if event_type == "start":
            agent["unique_tools"].add(tool_name)
            agent["unique_tool_count"] = len(agent["unique_tools"])
        elif event_type == "end":
            agent["tools_called"].append((tool_name, self._get_timestamp(), result))
            agent["tool_count"] = len(agent["tools_called"])

        self._print_debug_state(f"AFTER TOOL {event_type.upper()}: {agent_name} -> {tool_name}")

    def _record_error_event(self, error_type: str, agent_name: str, error_message: str, error_obj: Exception = None, context: dict = None):
        """Record detailed error event in unified dict"""
        error_event = {
            "event_id": self.event_counter,
            "timestamp": self._get_timestamp(),
            "error_type": error_type,
            "agent_name": agent_name,
            "error_message": error_message,
            "error_obj": error_obj,
            "context": context or {},
            "agent_state_at_error": self.workflow_events["agents"].get(agent_name, {}).copy() if agent_name in self.workflow_events["agents"] else {}
        }
        
        # Add to error_events list
        self.workflow_events["error_events"].append(error_event)
        
        # Also add to legacy errors list for compatibility
        self.workflow_events["errors"].append({
            "timestamp": self._get_timestamp(),
            "error_type": error_type,
            "error": error_message,
            "agent": agent_name,
            "event_id": self.event_counter
        })
        
    def _record_handoff_event(self, from_agent: str, to_agent: str):
        """Record handoff event in unified dict"""
        self._ensure_agent_exists(from_agent)
        self._ensure_agent_exists(to_agent)
        
        handoff_event = {
            "event_id": self.event_counter,
            "timestamp": self._get_timestamp(),
            "from_agent": from_agent,
            "to_agent": to_agent
        }
        
        # Add to global handoffs list
        self.workflow_events["handoffs"].append(handoff_event)
        
        # Update agent-specific tracking
        self.workflow_events["agents"][from_agent]["handoffs_from"].append(to_agent)
        self.workflow_events["agents"][to_agent]["handoffs_to"].append(from_agent)

    def _record_output_event(self, agent_name: str, output):
        """Record agent output in unified dict"""
        self._ensure_agent_exists(agent_name)
        
        output_event = {
            "event_id": self.event_counter,
            "timestamp": self._get_timestamp(),
            "agent_name": agent_name,
            "output": str(output),
            "output_obj": output
        }
        
        # Add to global outputs list
        self.workflow_events["outputs"].append(output_event)
        
        # Update agent-specific tracking
        self.workflow_events["agents"][agent_name]["outputs"].append(output)

    def _check_agent_ready(self, agent_name: str, raise_exceptions: bool = False) -> tuple[bool, list[str]]:
        """Check if agent meets requirements using unified dict. Optionally raise specific exceptions."""
        if agent_name not in self.workflow_events["agents"]:
            error_msg = f"{agent_name} not found in tracking"
            if raise_exceptions:
                raise AgentStateViolationError(agent_name, error_msg, context={"workflow_events": self.workflow_events})
            return False, [error_msg]
        
        agent = self.workflow_events["agents"][agent_name]
        requirements = agent["requirements"]
        errors = []
        
        # Check minimum tools
        min_tools = requirements.get("min_tools", 0)
        if agent["unique_tool_count"] < min_tools:
            error_msg = f"Need {min_tools} unique tools, only called {agent['unique_tool_count']}"
            if raise_exceptions:
                raise PrematureCompletionError(
                    agent_name, 
                    [f"minimum {min_tools} unique tools"],
                    context={
                        "current_tool_count": agent["unique_tool_count"],
                        "min_required": min_tools,
                        "tools_called": list(agent["unique_tools"])
                    }
                )
            errors.append(error_msg)
        
        # Check required tools
        required_tools = requirements.get("required_tools", [])
        missing_tools = [tool for tool in required_tools if tool not in agent["unique_tools"]]
        if missing_tools:
            if raise_exceptions:
                raise RequiredToolNotCalledError(
                    agent_name,
                    missing_tools,
                    context={
                        "required_tools": required_tools,
                        "tools_called": list(agent["unique_tools"]),
                        "agent_state": agent
                    }
                )
            errors.append(f"Missing required tools: {missing_tools}")
        
        return len(errors) == 0, errors

    def _check_valid_handoff(self, from_agent: str, to_agent: str) -> tuple[bool, str]:
        """Check if handoff is valid using unified dict"""
        if from_agent not in self.workflow_events["agents"]:
            return False, f"{from_agent} not found"
        
        requirements = self.workflow_events["agents"][from_agent]["requirements"]
        valid_targets = requirements.get("valid_targets", [])
        
        if valid_targets and to_agent not in valid_targets:
            return False, f"Can only handoff to: {valid_targets}"
        
        return True, ""

    def _print_debug_state(self, context: str = ""):
        """Print detailed current state for debugging - called after every tool event"""
        print(f"\n{'🔍'*10} DEBUG STATE {context} {'🔍'*10}")
        
        # Show all agents and their current status
        for agent_name, agent in self.workflow_events["agents"].items():
            print(f"\n🤖 {agent_name}:")
            print(f"   🔧 Unique tools ({agent['unique_tool_count']}): {sorted(list(agent['unique_tools']))}")
            print(f"   📊 Total tool calls: {agent['tool_count']}")
            
            # Show recent tool calls with results
            if agent["tools_called"]:
                recent = agent["tools_called"][-3:]  # Last 3 calls
                recent_calls = [f"{tool}({len(result or '')} chars)" for tool, _, result in recent]
                print(f"   📝 Recent calls: {recent_calls}")
                
            # Show handoff activity
            if agent['handoffs_from'] or agent['handoffs_to']:
                print(f"   🔄 Handoffs: from→{agent['handoffs_from']} to←{agent['handoffs_to']}")
            
            # Check and show readiness
            is_ready, errors = self._check_agent_ready(agent_name)
            requirements = agent["requirements"]
            
            print(f"   📋 Requirements:")
            print(f"      • Min tools: {requirements.get('min_tools', 0)} (current: {agent['unique_tool_count']})")
            if requirements.get('required_tools'):
                missing = [t for t in requirements['required_tools'] if t not in agent['unique_tools']]
                print(f"      • Required tools: {requirements['required_tools']}")
                if missing:
                    print(f"      • ❌ Missing: {missing}")
                else:
                    print(f"      • ✅ All required tools called")
            
            print(f"   🚦 Status: {'✅ READY' if is_ready else '❌ NOT READY'}")
            if not is_ready:
                print(f"      Reasons: {', '.join(errors)}")
        
        # Show global workflow stats
        print(f"\n📈 WORKFLOW TOTALS:")
        print(f"   🔧 Tool events: {len(self.workflow_events['tools'])} (last: {self.workflow_events['tools'][-1]['tool_name'] if self.workflow_events['tools'] else 'None'})")
        print(f"   🔄 Handoffs: {len(self.workflow_events['handoffs'])}")
        print(f"   📤 Outputs: {len(self.workflow_events['outputs'])}")
        print(f"   🚨 Errors: {len(self.workflow_events['errors'])}")
        print(f"   📋 Error Events: {len(self.workflow_events['error_events'])}")
        
        # Show recent tool timeline
        if len(self.workflow_events['tools']) > 0:
            print(f"\n📅 RECENT TOOL TIMELINE:")
            recent_tools = self.workflow_events['tools'][-5:]  # Last 5 tool events
            for i, tool_event in enumerate(recent_tools):
                event_icon = "🟢" if tool_event['event_type'] == 'start' else "🔴"
                print(f"   {i+1}. {event_icon} {tool_event['agent_name']} -> {tool_event['tool_name']} ({tool_event['event_type']})")
        
        print(f"{'🔍'*50}\n")
    
    def _print_current_state(self, context: str = ""):
        """Print current state of all agents from unified dict"""
        print(f"\n{'='*10} CURRENT STATE {context} {'='*10}")
        
        for agent_name, agent in self.workflow_events["agents"].items():
            print(f"\n🤖 {agent_name}:")
            print(f"   🔧 Unique tools: {agent['unique_tool_count']} {sorted(list(agent['unique_tools']))}")
            print(f"   📊 Total tool calls: {agent['tool_count']}")
            print(f"   🔄 Handoffs from: {len(agent['handoffs_from'])}")
            print(f"   📤 Outputs: {len(agent['outputs'])}")
            
            # Check readiness
            is_ready, errors = self._check_agent_ready(agent_name)
            if is_ready:
                print(f"   ✅ Ready for handoff")
            else:
                print(f"   ❌ Not ready: {', '.join(errors)}")
        
        print(f"\n📈 WORKFLOW TOTALS:")
        print(f"   🔧 Total tool events: {len(self.workflow_events['tools'])}")
        print(f"   🔄 Total handoffs: {len(self.workflow_events['handoffs'])}")
        print(f"   📤 Total outputs: {len(self.workflow_events['outputs'])}")
        print(f"   🚨 Total errors: {len(self.workflow_events['errors'])}")
        print(f"   📋 Total error events: {len(self.workflow_events['error_events'])}")
        print(f"{'='*40}\n")

    # ========================================================================
    # HOOK METHODS USING UNIFIED TRACKING
    # ========================================================================

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        """Initialize agent in unified tracking"""
        print (f'in on_agent_start agent.name {agent.name} context {context}')
        self.event_counter += 1
        self._safe_add_tokens(context)
        
        self._ensure_agent_exists(agent.name)
        
        print(f"\n{'='*15} AGENT START {'='*15}")
        print(f"### {self.event_counter}: Agent {agent.name} started")
        
        # Show requirements from unified dict
        agent_data = self.workflow_events["agents"][agent.name]
        requirements = agent_data["requirements"]
        if requirements.get("required_tools"):
            print(f"📋 Required tools: {requirements['required_tools']}")
        if requirements.get("min_tools", 0) > 0:
            print(f"🔢 Minimum tools: {requirements['min_tools']}")
        
        self._print_current_state("AFTER AGENT START")

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        """Record agent completion in unified tracking"""
        print (f'in on_agent_end agent.name {agent.name} output {output} context {context}')
        self.event_counter += 1
        self._safe_add_tokens(context)
        
        # Record output
        self._record_output_event(agent.name, output)
        
        print(f"\n{'='*15} AGENT END {'='*15}")
        print(f"### {self.event_counter}: Agent {agent.name} completed")
        
        # Show agent status from unified dict
        if agent.name in self.workflow_events["agents"]:
            agent_data = self.workflow_events["agents"][agent.name]
            print(f"\n🔧 FINAL TOOL STATUS:")
            print(f"   📊 Unique tools: {agent_data['unique_tool_count']}")
            print(f"   📈 Total calls: {agent_data['tool_count']}")
            print(f"   🛠️  Tools: {sorted(list(agent_data['unique_tools']))}")
            
            # Show tool call details
            if agent_data["tools_called"]:
                recent_tools = [tool_name for tool_name, _, _ in agent_data["tools_called"][-5:]]
                print(f"   📝 Recent calls: {recent_tools}")
        
        self._print_current_state("AFTER AGENT END")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        """Record tool start in unified tracking"""
        print (f'in on_tool_start agent.name {agent.name} tool.name {tool.name} context {context}')
        self.event_counter += 1
        self._safe_add_tokens(context)
        
        # Record tool start
        self._record_tool_event(agent.name, tool.name, "start")
        
        print(f"\n{'='*15} TOOL START {'='*15}")
        print(f"### {self.event_counter}: Tool {tool.name} started by {agent.name}")
        
        # Check current state after tool start
        is_ready, errors = self._check_agent_ready(agent.name)
        if agent.name in self.workflow_events["agents"]:
            agent_data = self.workflow_events["agents"][agent.name]
            print(f"🔧 Current tools: {sorted(list(agent_data['unique_tools']))}")
            print(f"📊 Unique count: {agent_data['unique_tool_count']}")
            if not is_ready:
                print(f"⚠️  Still needed: {', '.join(errors)}")

    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str) -> None:
        """Record tool completion in unified tracking"""
        print (f'in on_tool_end agent.name {agent.name} tool.name {tool.name} result {result} context {context}')
        self.event_counter += 1
        self._safe_add_tokens(context)
        
        # Record tool end with result
        self._record_tool_event(agent.name, tool.name, "end", result)
        
        print(f"\n{'='*15} TOOL END {'='*15}")
        print(f"### {self.event_counter}: Tool {tool.name} completed for {agent.name}")
        print(f"📤 Result length: {len(result)} characters")
        
        # Show updated status from unified dict
        if agent.name in self.workflow_events["agents"]:
            agent_data = self.workflow_events["agents"][agent.name]
            print(f"\n🔧 UPDATED TOOL STATUS:")
            print(f"   📊 Unique tools: {agent_data['unique_tool_count']}")
            print(f"   📈 Total calls: {agent_data['tool_count']}")
            print(f"   🛠️  All tools: {sorted(list(agent_data['unique_tools']))}")
            
            # Check readiness
            is_ready, errors = self._check_agent_ready(agent.name)
            if is_ready:
                print(f"   ✅ All requirements met!")
                agent_data["ready_for_handoff"] = True
            else:
                print(f"   ⚠️  Still needed: {', '.join(errors)}")

    async def on_handoff(self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent) -> None:
        """Validate and record handoff using unified tracking"""
        print (f'in on_handoff from_agent.name {from_agent.name} to_agent.name {to_agent.name} context {context}')
        self.event_counter += 1
        self._safe_add_tokens(context)
        
        print(f"\n{'='*15} HANDOFF VALIDATION {'='*15}")
        print(f"### {self.event_counter}: Validating {from_agent.name} → {to_agent.name}")
        
        # Validate using unified dict
        is_ready, readiness_errors = self._check_agent_ready(from_agent.name)
        is_valid_target, target_error = self._check_valid_handoff(from_agent.name, to_agent.name)
        
        # Show current state before validation
        self._print_current_state("BEFORE HANDOFF")
        
        # Use exception-raising version for validation failures
        if not is_ready:
            try:
                # This will raise the appropriate specific exception
                self._check_agent_ready(from_agent.name, raise_exceptions=True)
            except (RequiredToolNotCalledError, PrematureCompletionError, AgentStateViolationError) as error:
                # Record detailed error event
                self._record_error_event(
                    error_type=error.__class__.__name__,
                    agent_name=from_agent.name,
                    error_message=str(error),
                    error_obj=error,
                    context={
                        "validation_context": "handoff_validation",
                        "attempted_handoff_to": to_agent.name,
                        "unified_state": self.workflow_events
                    }
                )
                print(f"🚨 VALIDATION FAILED: {error}")
                raise error
        
        if not is_valid_target:
            error = InvalidHandoffError(
                from_agent.name,
                to_agent.name,
                target_error,
                context={"unified_state": self.workflow_events}
            )
            # Record detailed error event
            self._record_error_event(
                error_type=error.__class__.__name__,
                agent_name=from_agent.name,
                error_message=str(error),
                error_obj=error,
                context={
                    "validation_context": "handoff_target_validation",
                    "attempted_handoff_to": to_agent.name,
                    "valid_targets": self.workflow_events["agents"][from_agent.name]["requirements"].get("valid_targets", [])
                }
            )
            print(f"🚨 VALIDATION FAILED: {error}")
            raise error
        
        # Record successful handoff
        self._record_handoff_event(from_agent.name, to_agent.name)
        
        print(f"✅ HANDOFF VALIDATED: {from_agent.name} → {to_agent.name}")
        print(f"{'='*50}\n")

    # ========================================================================
    # UNIFIED REPORTING
    # ========================================================================

    def get_unified_summary(self):
        """Get complete workflow summary from unified dict"""
        return {
            "execution_stats": {
                "total_events": self.event_counter,
                "total_agents": len(self.workflow_events["agents"]),
                "total_tools": len(self.workflow_events["tools"]),
                "total_handoffs": len(self.workflow_events["handoffs"]),
                "total_outputs": len(self.workflow_events["outputs"]),
                "total_errors": len(self.workflow_events["errors"]),
                "total_error_events": len(self.workflow_events["error_events"]),
                "token_usage": {
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens
                }
            },
            "complete_workflow": self.workflow_events
        }

    def save_workflow_log(self, filename="unified_workflow.json"):
        """Save unified workflow log"""
        summary = self.get_unified_summary()
        import json
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"💾 Unified workflow log saved to {filename}")
        return filename

    def print_final_summary(self):
        """Print comprehensive final summary"""
        print(f"\n{'='*20} FINAL WORKFLOW SUMMARY {'='*20}")
        
        # Agent summaries
        for agent_name, agent in self.workflow_events["agents"].items():
            print(f"\n🤖 {agent_name}:")
            print(f"   🔧 Tools called: {sorted(list(agent['unique_tools']))}")
            print(f"   📊 Tool counts: {agent['unique_tool_count']} unique, {agent['tool_count']} total")
            print(f"   🔄 Handoffs: {len(agent['handoffs_from'])} outgoing, {len(agent['handoffs_to'])} incoming")
            print(f"   📤 Outputs: {len(agent['outputs'])}")
        
        # Workflow flow
        print(f"\n🔄 HANDOFF CHAIN:")
        for handoff in self.workflow_events["handoffs"]:
            print(f"   {handoff['from_agent']} → {handoff['to_agent']}")
        
        # Tool usage
        print(f"\n🔧 TOOL USAGE:")
        all_tools = {}
        for tool_event in self.workflow_events["tools"]:
            tool_key = f"{tool_event['agent_name']}-{tool_event['tool_name']}"
            all_tools[tool_key] = all_tools.get(tool_key, 0) + 1
        for tool_key, count in sorted(all_tools.items()):
            print(f"   {tool_key}: {count} times")
        
        # Error summary
        if self.workflow_events["error_events"]:
            print(f"\n🚨 ERROR EVENTS:")
            for error_event in self.workflow_events["error_events"]:
                print(f"   {error_event['timestamp']}: {error_event['error_type']} - {error_event['agent_name']}")
                print(f"      Message: {error_event['error_message']}")
        
        print(f"{'='*60}\n")

    def get_error_events(self):
        """Get detailed error events for analysis"""
        return self.workflow_events["error_events"]

    def get_error_summary(self):
        """Get error analysis summary"""
        error_events = self.workflow_events["error_events"]
        
        if not error_events:
            return {"total_error_events": 0, "error_types": {}, "agents_with_errors": {}}
        
        error_types = {}
        agents_with_errors = {}
        
        for error_event in error_events:
            error_type = error_event["error_type"]
            agent_name = error_event["agent_name"]
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            agents_with