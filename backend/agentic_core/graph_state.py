# backend/agentic_core/graph_state.py
from __future__ import annotations

import os
import json
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel

class AgentAction(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: Optional[Union[str, dict]] = None
    reasoning: Optional[str] = None

class AgentState(BaseModel):
    input: str
    chat_history: List[dict]
    intermediate_steps: List[AgentAction]
    agent_outcome: Union[str, dict, None] = None
    plan: List[str]
    metadata: Dict[str, Union[str, int, float]]
    self_refine_iter: int
    context_cache: Dict[str, Any]
    reflect_decision: Optional[str] = None

class CustomMemorySaver:
    def __init__(self, filename: str = "agent_memory.json"):
        self.filename = filename

    @property
    def config(self) -> dict:
        return {"filename": self.filename}

    def save(self, state: dict) -> None:
        def pydantic_to_dict(o):
            if isinstance(o, BaseModel):
                return o.model_dump() if hasattr(o, "model_dump") else o.dict()
            return str(o)
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        with open(self.filename, "w") as f:
            json.dump(state, f, default=pydantic_to_dict, indent=2)
        print(f"✅ State saved to {self.filename}")

    def load(self) -> Union[dict, None]:
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                return json.load(f)
        return None
