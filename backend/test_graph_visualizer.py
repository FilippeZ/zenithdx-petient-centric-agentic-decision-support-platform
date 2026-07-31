# backend/test_graph_visualizer.py
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipelines.graph_ehr.graph_visualizer import ZenithGraphVisualizer

def run_visualization_tests():
    print("=" * 70)
    print("[TEST] ZENITHDX GRAPH VISUALIZATION MODULE TEST")
    print("Testing PyVis HTML rendering & Gephi GraphML exporting...")
    print("=" * 70)

    visualizer = ZenithGraphVisualizer()

    # 1. Test Gephi GraphML Export
    print("\n1. Testing Gephi GraphML Export...")
    gephi_path = "zenithdx_full_graph.graphml"
    result_gephi = visualizer.export_to_gephi(gephi_path)
    assert os.path.exists(result_gephi), f"Gephi GraphML file missing at {result_gephi}"
    gephi_size = os.path.getsize(result_gephi)
    print(f"   * Gephi export verified: {result_gephi} ({gephi_size} bytes)")

    # 2. Test PyVis Interactive HTML Export
    print("\n2. Testing PyVis Interactive HTML Export...")
    html_path = "patient_subgraph.html"
    result_html = visualizer.export_patient_subgraph_pyvis(10000032, html_path)
    assert os.path.exists(result_html), f"PyVis HTML file missing at {result_html}"
    html_size = os.path.getsize(result_html)
    print(f"   * PyVis HTML export verified: {result_html} ({html_size} bytes)")

    # 3. Test Dynamic PyVis HTML String Generation (FastAPI endpoint backend)
    print("\n3. Testing Dynamic HTML String Generation for FastAPI Endpoint...")
    html_str = visualizer.generate_patient_subgraph_html_string(10000032)
    assert "<html" in html_str.lower() and "vis-network" in html_str.lower(), "Dynamic HTML output invalid"
    print(f"   * Dynamic HTML string generated successfully ({len(html_str)} chars)")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL GRAPH VISUALIZATION TESTS PASSED CLEANLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_visualization_tests()
