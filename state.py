# state.py
from __future__ import annotations

import operator
from typing import Annotated, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish

class Asset(TypedDict):
    asset_id: str
    path: str
    source_url: str

class AgentState(TypedDict):
    topic: str
    target_words: int                 
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    assets: Annotated[list[Asset], operator.add]
