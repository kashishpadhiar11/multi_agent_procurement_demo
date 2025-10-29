"""SupplierAgent
=================

Selects a preferred supplier for a requested item. For demo purposes this
agent uses a simple in-memory mapping and a keyword-based retriever built on
LangChain's `BaseRetriever`. This keeps dependencies light and avoids any
external services.

Example
-------
>>> mapping = {"laptop": "Acme Computers", "chair": "OfficeCo"}
>>> agent = SupplierAgent(item_to_supplier=mapping)
>>> agent.get_supplier("laptops")
{'item': 'laptops', 'supplier': 'Acme Computers'}
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda


class KeywordMappingRetriever(BaseRetriever):
    """A minimal keyword-based retriever over an item->supplier mapping.

    It returns a single `Document` when the item (after naive normalization)
    matches a known key. This is intentionally simple for demo clarity.
    """

    def __init__(self, item_to_supplier: Dict[str, str]):
        super().__init__()
        # Normalize keys to lowercase singular for lookup simplicity
        self._mapping = {self._normalize(k): v for k, v in item_to_supplier.items()}

    def _get_relevant_documents(self, query: str) -> List[Document]:
        key = self._normalize(query)
        supplier = self._mapping.get(key)
        if supplier:
            return [Document(page_content=supplier, metadata={"item": query})]
        # Fallback: try a naive plural->singular or singular->plural heuristic
        alt_key = self._alt_normalize(key)
        supplier = self._mapping.get(alt_key)
        if supplier:
            return [Document(page_content=supplier, metadata={"item": query})]
        return []

    def _normalize(self, text: str) -> str:
        text = text.strip().lower()
        if text.endswith("s") and not text.endswith("ss"):
            return text[:-1]
        return text

    def _alt_normalize(self, text: str) -> str:
        # Toggle simple plural form for a second chance match
        if text.endswith("s") and not text.endswith("ss"):
            return text[:-1]
        return f"{text}s"


class SupplierAgent:
    """Looks up a preferred supplier for a given item via a simple retriever.

    Public methods:
      - `get_supplier(item)`: returns a dict with `item` and `supplier` keys

    Also exposes a LangChain Runnable at `runnable` for chain composition.
    """

    def __init__(self, item_to_supplier: Optional[Dict[str, str]] = None) -> None:
        self.item_to_supplier: Dict[str, str] = item_to_supplier or {}
        self.retriever = KeywordMappingRetriever(self.item_to_supplier)
        self.runnable = RunnableLambda(self._lookup_to_dict)

    def get_supplier(self, item: str) -> Dict[str, str]:
        return self._lookup_to_dict(item)

    def _lookup_to_dict(self, item: str) -> Dict[str, str]:
        # In newer LangChain versions, retrievers are Runnables.
        # Use `invoke` instead of `get_relevant_documents`.
        docs = self.retriever.invoke(item)
        supplier = docs[0].page_content if docs else "Unknown Supplier"
        return {"item": item, "supplier": supplier}


__all__ = ["SupplierAgent", "KeywordMappingRetriever"]

