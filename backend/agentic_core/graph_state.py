# backend/agentic_core/graph_state.py
from __future__ import annotations

import os
import json
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field

class AgentAction(BaseModel):
    tool_name: str
    tool_input: dict = Field(default_factory=dict)
    tool_output: Optional[Union[str, dict]] = None
    reasoning: Optional[str] = None

class AgentState(BaseModel):
    input: str
    chat_history: List[dict] = Field(default_factory=list)
    intermediate_steps: List[AgentAction] = Field(default_factory=list)
    agent_outcome: Union[str, dict, None] = None
    plan: List[str] = Field(default_factory=list)
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict)
    self_refine_iter: int = 0
    step_count: int = 0  # Infinite loop protection guard
    context_cache: Dict[str, Any] = Field(default_factory=dict)
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
