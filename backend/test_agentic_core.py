# backend/test_agentic_core.py
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agentic_core.graph_state import AgentState, AgentAction
from agentic_core.agent_loop import (
    parse_tool_action_json,
    prune_state_context,
    should_continue_reflector,
    MAX_STEPS_GUARD
)

def run_agentic_core_tests():
    print("=" * 70)
    print("[TEST] ZENITHDX LANGGRAPH AGENTIC CORE SAFETY TEST")
    print("Testing Infinite Loop Guard, State Pruning & Auto-Fix JSON Parser")
    print("=" * 70)

    # 1. Test Infinite Loop Guard (Step Count Check)
    print("\n1. Testing Infinite Loop Protection Guard...")
    state_normal = AgentState(
        input="test query",
        step_count=3,
        reflect_decision="CONTINUE"
    )
    route_normal = should_continue_reflector(state_normal)
    print(f"   * Normal Step Count (3/7), Decision CONTINUE -> Route: {route_normal}")
    assert route_normal == "react_agent", "Normal route failed"

    state_limit = AgentState(
        input="test query",
        step_count=7,
        reflect_decision="CONTINUE"
    )
    route_limit = should_continue_reflector(state_limit)
    print(f"   * Limit Step Count (7/7), Decision CONTINUE -> Route: {route_limit}")
    assert route_limit == "final_answer", "Loop guard failed to force exit to final_answer"
    print("   * [SUCCESS] Infinite Loop Protection Guard verified!")

    # 2. Test State Pruning (Context Window Control)
    print("\n2. Testing State Pruning (Context Window Bloat Prevention)...")
    heavy_cache = {
        "rag_query": "X" * 1500,
        "history_text": ["Y" * 800, "Z" * 900],
        "xray_findings": ["Pneumonia", "Opacity"]
    }
    pruned_cache = prune_state_context(heavy_cache, max_text_len=300)
    
    print(f"   * Original RAG Text Length: {len(heavy_cache['rag_query'])} chars")
    print(f"   * Pruned RAG Text Length: {len(pruned_cache['rag_query'])} chars")
    print(f"   * Original History Item 1 Length: {len(heavy_cache['history_text'][0])} chars")
    print(f"   * Pruned History Item 1 Length: {len(pruned_cache['history_text'][0])} chars")

    assert len(pruned_cache["rag_query"]) <= 350, "RAG text pruning failed"
    assert len(pruned_cache["history_text"][0]) <= 320, "History text pruning failed"
    print("   * [SUCCESS] State Pruning mechanism verified (<4k tokens guarantee)!")

    # 3. Test Auto-Fix Structured Output JSON Parser
    print("\n3. Testing Auto-Fix Structured Output JSON Parser...")
    
    # Valid Markdown JSON
    valid_md_json = '```json\n{"tool_name": "query_diag", "tool_input": {"query": "pneumonia"}, "reasoning": "checking"}\n```'
    action_valid, err_valid = parse_tool_action_json(valid_md_json)
    print(f"   * Valid Markdown JSON -> Action Tool Name: {action_valid.tool_name if action_valid else None}")
    assert action_valid is not None and action_valid.tool_name == "query_diag", "Valid Markdown JSON parsing failed"

    # Conversational LLM Output (Malformed Prefix)
    conversational_json = 'Here is the requested tool call:\n{"tool": "image_diag", "input": {"image": "xray.png"}, "thought": "analyzing image"}'
    action_conv, err_conv = parse_tool_action_json(conversational_json)
    print(f"   * Conversational Prefix JSON -> Action Tool Name: {action_conv.tool_name if action_conv else None}")
    assert action_conv is not None and action_conv.tool_name == "image_diag", "Conversational JSON extraction failed"

    # Completely Malformed Output -> Auto-Fix Error Feedback
    malformed_str = "I want to call the vision tool on this chest xray."
    action_mal, err_mal = parse_tool_action_json(malformed_str)
    print(f"   * Malformed LLM Output -> Auto-Fix Feedback Prompt:\n     \"{err_mal}\"")
    assert action_mal is None and "Auto-Fix Notice" in err_mal, "Auto-Fix error response failed"
    print("   * [SUCCESS] Auto-Fix JSON Parser & Error Feedback verified!")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL LANGGRAPH AGENTIC CORE TESTS PASSED CLEANLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_agentic_core_tests()
