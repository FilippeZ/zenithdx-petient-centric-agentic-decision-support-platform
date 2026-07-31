# backend/export_agent_diagram.py
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agentic_core.agent_loop import runnable

MERMAID_THESIS_DIAGRAM = """---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([__start__]);
	planner(planner);
	react_agent(react_agent);
	run_tool(run_tool);
	reflector(reflector);
	final_answer(final_answer);
	__end__([__end__]);

	__start__ --> planner;
	planner --> react_agent;
	
	%% ReAct Reasoning & Tool Execution Loop
	react_agent -->|tool_calls| run_tool;
	run_tool --> react_agent;
	
	%% Observation Completion & Self-Refinement
	react_agent -->|no_tools| reflector;
	
	%% Self-Refine Loop & Infinite Loop Safety Guard
	reflector -.->|REVISE| planner;
	reflector -.->|CONTINUE / step_count >= 7| final_answer;
	
	%% Synthesis & Final Output Generation
	final_answer --> __end__;

	classDef default fill:#1e1e1e,stroke:#3a7bd5,stroke-width:1.5px,color:#ffffff;
	classDef first fill-opacity:0,stroke:#00d2ff,stroke-width:2px;
	classDef last fill-opacity:0,stroke:#00b09b,stroke-width:2px;
"""

def export_agent_graph_diagram():
    print("=" * 70)
    print("[EXPORT] ZENITHDX LANGGRAPH AGENT ARCHITECTURE DIAGRAM EXPORT")
    print("Generating Mermaid syntax and PNG diagram for thesis document...")
    print("=" * 70)

    mmd_filename = "zenithdx_agent_architecture.mmd"
    with open(mmd_filename, "w", encoding="utf-8") as f:
        f.write(MERMAID_THESIS_DIAGRAM)
    print(f"[OK] Saved full architecture Mermaid code to: {mmd_filename}")

    # Export PNG Diagram via LangGraph draw_mermaid_png() or fallback
    png_filename = "zenithdx_agent_architecture.png"
    try:
        graph_obj = runnable.get_graph()
        graph_png = graph_obj.draw_mermaid_png()
        with open(png_filename, "wb") as f:
            f.write(graph_png)
        print(f"[OK] Saved agent architecture PNG diagram to: {png_filename} ({len(graph_png)} bytes)")
    except Exception as e:
        print(f"[Notice] Direct PNG generation ({e}). Full Mermaid .mmd file is saved at {mmd_filename}.", file=sys.stderr)

    print("\n" + "=" * 70)
    print("[SUCCESS] EXPORT COMPLETE: Architecture diagram ready for thesis book!")
    print("=" * 70)
    return MERMAID_THESIS_DIAGRAM

if __name__ == "__main__":
    export_agent_graph_diagram()
