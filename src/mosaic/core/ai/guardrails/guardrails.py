

import asyncio
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import openai

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

class MessageOutput(BaseModel): 
    response: str

class GuardrailOutput(BaseModel): 
    reasoning: str

guardrail_agent = Agent( 
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=GuardrailFunctionOutput
)

@output_guardrail
async def reasoning_guardrail(ctx: RunContextWrapper, agent: Agent, output: MessageOutput) -> GuardrailFunctionOutput:
    """Guardrail for manager completion"""
    print(f"********************** Manager completion guardrail triggered: {output}")
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.reasoning
    )

