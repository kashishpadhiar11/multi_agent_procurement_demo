## Modular Procurement Multi-Agent Workflow (LangChain + LangGraph)

This demo showcases a small, fully self-contained procurement workflow built with:

- **IntakeAgent**: Parses natural language requests (e.g., "Order 3 laptops").
- **SupplierAgent**: Chooses a preferred supplier using a simple keyword retriever.
- **ApprovalAgent**: Applies a business policy (auto-approve if quantity ≤ 5).
- **LangGraph**: Orchestrates the agent nodes and passes state between them.

The goal is to provide a clear, beginner-friendly reference that runs without external LLMs or API keys.

### Why this design?
- **Modularity**: Each agent lives in its own file and exposes a clean API.
- **Explainability**: Lightweight, commented code with straightforward logic.
- **Orchestrated**: LangGraph stitches together the agent steps and state transitions.
- **Extensible**: Add new agents (e.g., ContractAgent) with minimal friction.

---

### Project Structure

```
multi_agent_procurement_demo/
├── agents/
│   ├── intake_agent.py       # Parses requests into {item, quantity}
│   ├── supplier_agent.py     # Selects preferred supplier via keyword retriever
│   └── approval_agent.py     # Applies policy: approve if quantity ≤ 5
│
├── workflow/
│   └── run_workflow.py       # LangGraph: build graph, run demo, print trace
│
├── examples/
│   └── test_requests.txt     # Example natural language requests
│
├── README.md                 # This file
└── requirements.txt          # Dependencies
```

---

### Agents Overview

- **IntakeAgent** (`agents/intake_agent.py`)
  - Uses a simple regex-based parser wrapped as a LangChain `Runnable`.
  - Output: `{ item: str, quantity: int, raw_request: str }`.

- **SupplierAgent** (`agents/supplier_agent.py`)
  - Implements a tiny keyword-based `BaseRetriever` over an in-memory mapping.
  - Output: `{ item: str, supplier: str }`.

- **ApprovalAgent** (`agents/approval_agent.py`)
  - Decision rule: approve if quantity ≤ 5, else manual approval required.
  - Output: `{ approved: bool, reason: str }`.

---

### Orchestration with LangGraph

The workflow state (a `TypedDict`) flows linearly:

1. `intake` node: parse request → `item`, `quantity`
2. `supplier` node: choose `supplier` by item
3. `approval` node: set `approved`, `reason`

Each node appends to a `logs` list for a readable end-to-end trace.

---

### Quickstart

1) Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run the workflow demo:

```bash
python -m workflow.run_workflow
```

You should see a trace similar to:

```
=== Procurement Workflow Trace ===
- Received request: Order 3 laptops
- Intake parsed: {'item': 'laptops', 'quantity': 3, 'raw_request': 'Order 3 laptops'}
- Supplier selected: {'item': 'laptops', 'supplier': 'Acme Computers'}
- Approval decision: {'approved': True, 'reason': 'Quantity 3 is within auto-approval threshold (<= 5).'}

=== Final Outcome ===
Item: laptops
Quantity: 3
Supplier: Acme Computers
Approved: True
Reason: Quantity 3 is within auto-approval threshold (<= 5).
```

The input is sourced from `examples/test_requests.txt`. You can add or reorder lines to try different cases (e.g., quantities > 5 to see disapproval).

---