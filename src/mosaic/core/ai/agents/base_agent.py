from typing import List, Optional
from pydantic import BaseModel, Field
from .base_prompt import BasePromptStructure
from pydantic import create_model

class AgentCreationPrompt:
    @classmethod
    def build(cls, role=None, job=None, rules=None, context=None, 
              inputs=None, outputs=None, good_examples=None, bad_examples=None, task=None):
        
        # Create new field definitions with updated descriptions
        field_definitions = {}
        
        # Get base field info
        base_fields = BasePromptStructure.model_fields
        
        # Create new fields with updated descriptions
        field_definitions['role'] = (str, Field(description=role if role else base_fields['role'].description))
        field_definitions['job'] = (str, Field(description=job if job else base_fields['job'].description))
        field_definitions['context'] = (Optional[str], Field(default=None, description=context if context else base_fields['context'].description))
        field_definitions['rules'] = (List[str], Field(default_factory=list, description=rules if rules else base_fields['rules'].description))
        field_definitions['inputs'] = (List[str], Field(default_factory=list, description=inputs if inputs else base_fields['inputs'].description))
        field_definitions['outputs'] = (List[str], Field(default_factory=list, description=outputs if outputs else base_fields['outputs'].description))
        field_definitions['good_examples'] = (List[str], Field(default_factory=list, description=good_examples if good_examples else base_fields['good_examples'].description))
        field_definitions['bad_examples'] = (List[str], Field(default_factory=list, description=bad_examples if bad_examples else base_fields['bad_examples'].description))
        field_definitions['task'] = (str, Field(description=task if task else base_fields['task'].description))
        # Create the new model class
        AgentCreationPrompt = create_model('AgentCreationPrompt', **field_definitions)
        
        return AgentCreationPrompt
    
    

class Step (BaseModel):
    step: str = Field(description="The step to take")
    reason: str = Field(description="The reason for the step")

class Plan (BaseModel):
    plan: List[Step] = Field(description="The steps to take")

class BaseAgent (BaseModel):
    name: str = Field(description="The name of the agent")
    instructions: str = Field(description="The instructions for the agent")
    functions: List[str] = Field(description="The functions that the agent must use, you must assign only one function per agent, you must use all agents and must pick the function only from the list of functions provided")
    reason: str = Field(description="The reason for the agent, and why the functions are highly relevant to the agent's objective")

class EvaluationCriteria(BaseModel):
    criteria: str = Field(description="The criteria to evaluate the output against")
    result: str = Field(description="The result of the evaluation")
    reason: str = Field(description="The reason for the evaluation")

class EvaluationResults(BaseModel):
    evaluation_criteria: List[EvaluationCriteria] = Field(description="The evaluation criteria and results")    


class EvaluatorAgent (BaseModel):
    name: str = Field(description="The name of the evaluator agent")
    instructions: str = Field(description="The instructions for the evaluator agent")
    evaluation_criteria: List[EvaluationResults] = Field(description="The evaluation criteria for the evaluator agent")

class ManagerAgent (BaseModel):
    name: str = Field(description="The name of the manager agent")
    instructions: str = Field(description="The instructions for the manager agent")
    functions: List[str] = Field(description="The functions that the manager agent must use, the only functions that the manager agent must use is plan_for_next_steps and generate_final_answer")

class BaseAgents (BaseModel):
        agents: List[BaseAgent] = Field(description="The agents to create")
        manager_agent: ManagerAgent = Field(description="The manager agent")
        evaluator_agent: EvaluatorAgent = Field(description="The evaluator agent")
        plan: Plan = Field(description="The plan for the agents")