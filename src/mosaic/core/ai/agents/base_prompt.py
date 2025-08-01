from pydantic import BaseModel, Field
from pydantic_ai import format_as_xml
from pydantic import BaseModel, Field
from pydantic_ai import format_as_xml
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum

# Core Models
class PromptComponent(BaseModel):
    """Base component for prompt elements"""
    name: str = Field(description="Name identifier for the component")
    description: str = Field(description="Detailed description of the component")
    required: bool = Field(default=True, description="Whether this component is required")

class Example(BaseModel):
    """Structure for examples"""
    input: str = Field(description="Input text for the example")
    output: str = Field(description="Expected output text for the example")
    explanation: Optional[str] = Field(default=None, description="Optional explanation of why this is a good/bad example")

class Rule(BaseModel):
    """Structure for rules with priority"""
    rule: str = Field(description="The rule or constraint to be followed")
    priority: int = Field(default=1, description="Priority level: 1=high, 2=medium, 3=low")
    context: Optional[str] = Field(default=None, description="Additional context or reasoning for the rule")

class EvaluationCriterion(BaseModel):
    """Structure for evaluation criteria"""
    criterion: str = Field(description="The evaluation criterion or metric")
    weight: float = Field(default=1.0, description="Weight/importance of this criterion in evaluation")
    measurement: str = Field(description="Type of measurement: binary, scale, count, etc.")

# Core Prompt Structure
class BasePromptStructure(BaseModel):
    """Base structure for all prompts"""
    role: str = Field(description="The role/persona for the AI")
    job: str = Field(description="The job description of the AI")
    task: str = Field(description="The task to accomplish")
    context: Optional[str] = Field(default=None, description="Additional context")
    
    # Core components
    inputs: List[PromptComponent] = Field(default_factory=list)
    outputs: List[PromptComponent] = Field(default_factory=list)
    
    # Learning components
    good_examples: List[Example] = Field(default_factory=list, description="List of positive examples showing desired behavior")
    bad_examples: List[Example] = Field(default_factory=list, description="List of negative examples showing what to avoid")
    
    # Constraint components
    rules: List[Rule] = Field(default_factory=list, description="List of rules and constraints to follow")
    check_list: List[str] = Field(default_factory=list, description="Checklist items to verify in the output")
    evaluation_criteria: List[EvaluationCriterion] = Field(default_factory=list, description="Criteria for evaluating the quality of output")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the prompt was created")
    version: str = Field(default="1.0", description="Version identifier for the prompt")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing and organizing prompts")