"""Run the modular procurement workflow with LangGraph.

This script wires together three modular agents using LangGraph:
  1) IntakeAgent     - parses natural language requests into structured data
  2) SupplierAgent   - selects a preferred supplier for the item
  3) ApprovalAgent   - applies business policy to auto-approve or not

It demonstrates state passing between nodes and prints a readable trace of
each step and the final outcome.

Usage
-----
Run from project root:
    python -m workflow.run_workflow

You can modify the example requests in `examples/test_requests.txt`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.intake_agent import IntakeAgent
from agents.supplier_agent import SupplierAgent
from agents.approval_agent import ApprovalAgent


class WorkflowState(TypedDict, total=False):
    """State passed between graph nodes.

    Keys:
        original_request: Original natural language input
        item: Parsed item noun
        quantity: Parsed integer quantity
        supplier: Preferred supplier name
        approved: Boolean approval decision
        reason: Human-readable reason for the decision
        logs: A list of textual log entries for traceability
    """

    original_request: str
    item: str
    quantity: int
    supplier: str
    approved: bool
    reason: str
    logs: List[str]


def build_graph() -> StateGraph:
    """Construct the LangGraph state graph with three nodes and linear edges."""

    graph = StateGraph(WorkflowState)

    intake_agent = IntakeAgent()
    supplier_agent = SupplierAgent(
        item_to_supplier={
            "laptop": "Acme Computers",
            "monitor": "Display World",
            "mouse": "Pointer Pros",
            "keyboard": "KeyCo",
            "chair": "OfficeCo",
            "desk": "FurnishIt",
        }
    )
    approval_agent = ApprovalAgent(max_auto_approve_quantity=5)

    def intake_node(state: WorkflowState) -> WorkflowState:
        request = state["original_request"]
        parsed = intake_agent.parse_request(request)
        logs = state.get("logs", []) + [f"Intake parsed: {parsed}"]
        return {
            **state,
            "item": parsed["item"],
            "quantity": int(parsed["quantity"]),
            "logs": logs,
        }

    def supplier_node(state: WorkflowState) -> WorkflowState:
        item = state.get("item", "")
        lookup = supplier_agent.get_supplier(item)
        logs = state.get("logs", []) + [f"Supplier selected: {lookup}"]
        return {
            **state,
            "supplier": lookup["supplier"],
            "logs": logs,
        }

    def approval_node(state: WorkflowState) -> WorkflowState:
        quantity = int(state.get("quantity", 1))
        decision = approval_agent.decide(quantity)
        logs = state.get("logs", []) + [f"Approval decision: {decision}"]
        return {
            **state,
            "approved": bool(decision["approved"]),
            "reason": str(decision["reason"]),
            "logs": logs,
        }

    graph.add_node("intake", intake_node)
    graph.add_node("supplier", supplier_node)
    graph.add_node("approval", approval_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "supplier")
    graph.add_edge("supplier", "approval")
    graph.add_edge("approval", END)

    return graph


def run_demo(request: str) -> Dict[str, object]:
    """Execute the workflow for a single request and print the trace."""

    graph = build_graph()
    app = graph.compile()

    initial_state: WorkflowState = {
        "original_request": request,
        "logs": [f"Received request: {request}"],
    }

    final_state: WorkflowState = app.invoke(initial_state)

    print("\n=== Procurement Workflow Trace ===")
    for entry in final_state.get("logs", []):
        print("-", entry)
    print("\n=== Final Outcome ===")
    print(f"Item: {final_state.get('item')}")
    print(f"Quantity: {final_state.get('quantity')}")
    print(f"Supplier: {final_state.get('supplier')}")
    print(f"Approved: {final_state.get('approved')}")
    print(f"Reason: {final_state.get('reason')}")

    return dict(final_state)


def main() -> None:
    # Use at least one example from examples/test_requests.txt
    try:
        with open("examples/test_requests.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        lines = ["Order 3 laptops"]

    for req in lines:  # Process all example requests
        _ = run_demo(req)


if __name__ == "__main__":
    main()
