"""ApprovalAgent
=================

Applies a simple business policy to determine whether a purchase request
should be approved. For demonstration purposes, the rule is:

    Approve if quantity <= 5, otherwise require manual approval.

This module is intentionally straightforward and self-contained. It exposes
the decision function as a LangChain Runnable for easy composition in a
LangGraph workflow.
"""

from __future__ import annotations

from typing import Dict

from langchain_core.runnables import RunnableLambda


class ApprovalAgent:
    """Encapsulates approval policy logic.

    Public API:
      - `decide(quantity)`: Returns a dict with `approved: bool` and `reason: str`.

    Also exposes `runnable` (a RunnableLambda) for use in chains/graphs.
    """

    def __init__(self, max_auto_approve_quantity: int = 5) -> None:
        self.max_auto_approve_quantity = max_auto_approve_quantity
        self.runnable = RunnableLambda(self._decide_to_dict)

    def decide(self, quantity: int) -> Dict[str, object]:
        return self._decide_to_dict(quantity)

    def _decide_to_dict(self, quantity: int) -> Dict[str, object]:
        if quantity <= self.max_auto_approve_quantity:
            return {
                "approved": True,
                "reason": f"Quantity {quantity} is within auto-approval threshold (<= {self.max_auto_approve_quantity}).",
            }
        return {
            "approved": False,
            "reason": f"Quantity {quantity} exceeds auto-approval threshold (> {self.max_auto_approve_quantity}).",
        }


__all__ = ["ApprovalAgent"]

