from __future__ import annotations

from typing import Literal

from langchain_core.agents import AgentFinish
from langgraph.graph import StateGraph, START, END

from state import AgentState
from nodes import reason_node, act_node


def should_continue(state: AgentState) -> Literal["act", END]:
    if isinstance(state.get("agent_outcome"), AgentFinish):
        return END
    return "act"


builder = StateGraph(AgentState)
builder.add_node("reason", reason_node)
builder.add_node("act", act_node)

builder.add_edge(START, "reason")
builder.add_conditional_edges("reason", should_continue)
builder.add_edge("act", "reason")

app = builder.compile()
