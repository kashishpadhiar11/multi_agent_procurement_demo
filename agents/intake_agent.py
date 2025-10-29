"""IntakeAgent
=================

Parses natural language procurement requests (e.g., "Order 3 laptops")
to extract the requested item and quantity. This module uses LangChain's
Runnable primitives to expose the parsing as a chain-compatible component
without requiring any external LLMs or APIs.

Design goals:
- Beginner-friendly implementation with clear docstrings and comments
- Self-contained: works offline without an API key
- LangChain-native: exposed as a Runnable for easy composition in chains/graphs

Example
-------
>>> agent = IntakeAgent()
>>> agent.parse_request("Order 3 laptops")
{'item': 'laptops', 'quantity': 3, 'raw_request': 'Order 3 laptops'}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from langchain_core.runnables import RunnableLambda


@dataclass
class IntakeResult:
    """Structured result produced by the `IntakeAgent`.

    Attributes:
        item: Normalized item noun (lowercase, plural form if applicable)
        quantity: Parsed integer quantity, defaults to 1 if not specified
        raw_request: The original natural language request
    """

    item: str
    quantity: int
    raw_request: str


class IntakeAgent:
    """Parses a natural language procurement request into structured data.

    The agent exposes a `runnable` compatible with LangChain's LCEL to allow
    composition inside larger chains or graphs. Internally, it uses a simple
    regex and some light normalization rules:

    - Quantity: the first integer it finds (defaults to 1 if none found)
    - Item: the remaining non-numeric words, normalized to lowercase
    - Basic plural handling: if quantity == 1, keep singular; otherwise, keep
      as-is (we do not attempt sophisticated lemmatization for simplicity).
    """

    def __init__(self) -> None:
        # Expose the parsing function as a LangChain Runnable for easy wiring.
        self.runnable = RunnableLambda(self._parse_to_dict)

    # Public API -----------------------------------------------------------
    def parse_request(self, request: str) -> Dict[str, object]:
        """Parse a natural language request into a dict.

        Args:
            request: Natural language procurement request (e.g., "Order 3 laptops").

        Returns:
            A dictionary with keys: `item`, `quantity`, `raw_request`.
        """
        return self._parse_to_dict(request)

    # Internal helpers -----------------------------------------------------
    def _parse_to_dict(self, request: str) -> Dict[str, object]:
        quantity = self._extract_quantity(request)
        item = self._extract_item(request, quantity)
        result = IntakeResult(item=item, quantity=quantity, raw_request=request)
        return {
            "item": result.item,
            "quantity": result.quantity,
            "raw_request": result.raw_request,
        }

    def _extract_quantity(self, request: str) -> int:
        match = re.search(r"(\d+)", request)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return 1

    def _extract_item(self, request: str, quantity: Optional[int]) -> str:
        # Remove numbers and common verbs like order/buy/purchase
        cleaned = re.sub(r"\d+", " ", request, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(order|buy|purchase|get|acquire|request)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

        # A very naive singular-plural normalization: if quantity == 1, try singular.
        # This is intentionally simple for demo purposes.
        if (quantity or 1) == 1:
            if cleaned.endswith("s") and not cleaned.endswith("ss"):
                cleaned = cleaned[:-1]
        return cleaned


__all__ = ["IntakeAgent", "IntakeResult"]

