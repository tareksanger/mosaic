
from typing import Any
from pydantic import BaseModel
from agents import (
    Agent, Runner, RunContextWrapper, input_guardrail, output_guardrail,
    GuardrailFunctionOutput, TResponseInputItem, RunHooks,
    InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, handoff
)
from agents import Agent, RunContextWrapper, RunHooks, Runner, Tool, Usage, function_tool, ModelSettings
from agents import WebSearchTool
import json

from ...agents.services.research.tools.serp_tools import search_news, search_organic

class NegotiationOffer(BaseModel):
    price: float
    message: str
    final_deal: bool = False

class NegotiationData(BaseModel):
    current_offer: float
    message: str

# ============================================================================
# TOOLS
# ============================================================================

@function_tool
async def search_topic(topic: str) -> str:
    """
    Search for organic results about a given topic including price of the product.
    
    Args:
        topic: The topic to search for  (e.g. "price of iphone 15")
        
    Returns:
        str: The organic search results
    """
    results = search_organic(str(topic))
    return results


# Input Guardrail - monitors all inputs to agents
@input_guardrail
async def input_context_logger(
    ctx: RunContextWrapper[None], 
    agent: Agent, 
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    print(f"\n{'='*20} INPUT GUARDRAIL {'='*20}")
    print(f"🤖 Agent: {agent.name}")
    print(f"📝 Input: {input}")
    print(f"🗂️  Context:")
    try:
        print(json.dumps(ctx.context, indent=2, default=str))
    except:
        print(ctx.context)
    print(f"{'='*60}\n")
    
    return GuardrailFunctionOutput(
        output_info=input,
        tripwire_triggered=False,
    )


@output_guardrail
async def output_context_logger(
    ctx: RunContextWrapper, 
    agent: Agent, 
    output: NegotiationOffer
) -> GuardrailFunctionOutput:
    print(f"\n{'='*20} OUTPUT GUARDRAIL {'='*20}")
    print(f"🤖 Agent: {agent.name}")
    print(f"💰 Price: ${output.price}")
    print(f"💬 Message: {output.message}")
    print(f"🤝 Final Deal: {output.final_deal}")
    print(f"🗂️  Context:")
    try:
        print(json.dumps(ctx.context, indent=2, default=str))
    except:
        print(ctx.context)
    print(f"{'='*60}\n")
    
    # Force handoff behavior: If this is buyer or seller agent, trigger handoff
    if agent.name in ["Buyer Agent", "Seller Agent"] and not output.final_deal:
        print(f"🔄 OUTPUT GUARDRAIL: Forcing handoff to orchestrator for {agent.name}")
        
        # Trigger handoff by modifying the output or context
        # This ensures the agent will make the handoff call
        return GuardrailFunctionOutput(
            output_info=output,
            tripwire_triggered=True,  # This will cause a specific behavior
        )
    
    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=False,
    )


# Output Guardrail - monitors all outputs from agents AND forces handoffs
@output_guardrail
async def output_context_logger_with_handoff(
    ctx: RunContextWrapper, 
    agent: Agent, 
    output: NegotiationOffer
) -> GuardrailFunctionOutput:
    print(f"\n{'='*20} OUTPUT GUARDRAIL {'='*20}")
    print(f"🤖 Agent: {agent.name}")
    print(f"💰 Price: ${output.price}")
    print(f"💬 Message: {output.message}")
    print(f"🤝 Final Deal: {output.final_deal}")
    print(f"🗂️  Context:")
    try:
        print(json.dumps(ctx.context, indent=2, default=str))
    except:
        print(ctx.context)
    print(f"{'='*60}\n")
    
    # Force handoff behavior: If this is buyer or seller agent, trigger handoff
    if agent.name in ["Buyer Agent", "Seller Agent"] and not output.final_deal:
        print(f"🔄 OUTPUT GUARDRAIL: Forcing handoff to orchestrator for {agent.name}")
        
        # Trigger handoff by modifying the output or context
        # This ensures the agent will make the handoff call
        return GuardrailFunctionOutput(
            output_info=output,
            tripwire_triggered=True,  # This will cause a specific behavior
        )
    
    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=False,
    )


# Add this guardrail: only orchestrator is validated
@output_guardrail
async def orchestrator_final_check(ctx, agent, output: NegotiationOffer) -> GuardrailFunctionOutput:
    if agent.name == "Negotiation Orchestrator" and not output.final_deal:
        return GuardrailFunctionOutput(
            output_info={"error": "Negotiation did not finish as final_deal=True"},
            tripwire_triggered=True
        )
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

@output_guardrail
async def seller_buyer_handoff_enforcer(ctx, agent, output: GuardrailFunctionOutput) -> GuardrailFunctionOutput:
    if agent.name in ["Seller Agent", "Buyer Agent"] and not output.final_deal:
        return GuardrailFunctionOutput(
            output_info={"error": f"{agent.name} exited without handoff"},
            tripwire_triggered=True
        )
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)