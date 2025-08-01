from agents import Agent as OpenAIAgent
from agents import Runner as OpenAIRunner
from agents import function_tool as openai_function_tool
from agents import  RawResponsesStreamEvent, TResponseInputItem, trace 
from agents import MessageOutputItem, HandoffOutputItem, ToolCallItem, ToolCallOutputItem, ItemHelpers
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from agents import Agent as OpenAIAgent
from agents import Runner as OpenAIRunner
from agents import function_tool as openai_function_tool
from agents import trace
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import contextlib
from enum import Enum

class AgentType(str, Enum):
    OPENAI = "openai"

class MosaicAgent:
    """Simple wrapper that creates OpenAI agents from BaseAgents with proper handoffs"""
    
    def __init__(self, base_agents, functions_dict=None):
        """
        Initialize MosaicAgent from BaseAgents structure
        
        Args:
            base_agents: BaseAgents instance containing agents, manager_agent, evaluator_agent, and plan
            functions_dict: Dictionary mapping function names to actual function objects
        """
        # Validate input
        if not hasattr(base_agents, 'agents'):
            raise ValueError(f"Expected BaseAgents instance, got {type(base_agents)}. "
                           "Make sure to pass an instance, not the class itself.")
        
        self.base_agents = base_agents
        self.functions_dict = functions_dict or {}
        self.agents = []
        self.manager_agent = None
        self.runner = OpenAIRunner()
        
        # Create the multi-agent system
        self._create_agents()
        self._setup_handoffs()
    
    def _create_function_tools(self, function_names):
        """Convert function names to OpenAI function tools"""
        tools = []
        for func_name in function_names:
            if func_name in self.functions_dict:
                func_obj = self.functions_dict[func_name]
                tool = openai_function_tool(func_obj)
                tools.append(tool)
            else:
                print(f"Warning: Function '{func_name}' not found in functions_dict")
        return tools
    
    def _create_agents(self):
        """Create OpenAI agents from BaseAgents"""
        
        # Create worker agents
        for base_agent in self.base_agents.agents:
            # Get function tools for this agent
            function_tools = self._create_function_tools(base_agent.functions)
            
            # Add handoff instruction
            handoff_instruction = f"""
            
After completing your tasks, hand off to the manager agent by calling the appropriate handoff function.
Only hand off after you have executed all your required functions: {base_agent.functions}
            """
            
            # Create OpenAI agent
            openai_agent = OpenAIAgent(
                name=base_agent.name,
                instructions=base_agent.instructions + handoff_instruction,
                functions=function_tools,
                model="gpt-4o-mini"
            )
            
            self.agents.append(openai_agent)
        
        # Create manager agent
        manager_base = self.base_agents.manager_agent
        manager_function_tools = self._create_function_tools(manager_base.functions)
        
        # Add plan context to manager instructions
        plan_context = ""
        if hasattr(self.base_agents, 'plan') and self.base_agents.plan:
            plan_steps = [f"{i+1}. {step.step}" for i, step in enumerate(self.base_agents.plan.plan)]
            plan_context = f"""
            
Your plan to follow:
{chr(10).join(plan_steps)}
            """
        
        self.manager_agent = OpenAIAgent(
            name=manager_base.name,
            instructions=manager_base.instructions + plan_context,
            functions=manager_function_tools,
            model="gpt-4o-mini"
        )
    
    def _setup_handoffs(self):
        """Setup handoff relationships between agents"""
        
        # Worker agents can only hand off to manager
        for agent in self.agents:
            agent.handoffs = [self.manager_agent]
        
        # Manager can hand off to all worker agents
        self.manager_agent.handoffs = self.agents.copy()
        
        # Add manager to agents list for completeness
        self.agents.append(self.manager_agent)
    
    def run(self, prompt, trace_name=None, **kwargs):
        """
        Run the multi-agent system starting with the manager
        
        Args:
            prompt: Initial prompt/objective for the system
            trace_name: Optional name for tracing
            **kwargs: Additional arguments passed to the runner
        
        Returns:
            Result from the agent execution
        """
        
        # Create input items
        if isinstance(prompt, str):
            input_items = [{"content": prompt, "role": "user"}]
        else:
            input_items = prompt
        
        # Use trace if provided
        trace_context = trace(trace_name or "mosaic_agent_run") if trace_name else contextlib.nullcontext()
        
        try:
            with trace_context:
                # Run synchronously
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If in async context, create task
                        return asyncio.create_task(
                            self.runner.run(self.manager_agent, input_items, **kwargs)
                        )
                    else:
                        # Run in existing loop
                        return loop.run_until_complete(
                            self.runner.run(self.manager_agent, input_items, **kwargs)
                        )
                except RuntimeError:
                    # No event loop, create new one
                    return asyncio.run(
                        self.runner.run(self.manager_agent, input_items, **kwargs)
                    )
        except Exception as e:
            print(f"Error running MosaicAgent: {e}")
            raise
    
    async def arun(self, prompt, trace_name=None, **kwargs):
        """
        Async version of run
        
        Args:
            prompt: Initial prompt/objective for the system
            trace_name: Optional name for tracing
            **kwargs: Additional arguments passed to the runner
        
        Returns:
            Result from the agent execution
        """
        
        # Create input items
        if isinstance(prompt, str):
            input_items = [{"content": prompt, "role": "user"}]
        else:
            input_items = prompt
        
        # Use trace if provided
        trace_context = trace(trace_name or "mosaic_agent_arun") if trace_name else contextlib.nullcontext()
        
        with trace_context:
            return await self.runner.run(self.manager_agent, input_items, **kwargs)
    
    def get_agent_by_name(self, name):
        """Get an agent by name"""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    @classmethod
    def from_config_data(cls, config_data, functions_dict=None):
        """
        Create MosaicAgent from the configuration data in your paste.txt
        
        Args:
            config_data: The BaseAgents data structure from your paste.txt
            functions_dict: Dictionary mapping function names to actual function objects
        """
        return cls(config_data, functions_dict)
    
    def list_agents(self):
        """List all agent names"""
        return [agent.name for agent in self.agents]

