# Multi-Agent Scattering Analysis System — Design Document

> **Status:** Design / proposal. Nothing here is implemented yet. This document
> describes a proposed architecture for extending the existing `sas_llm/` LLM
> workflow into a multi-agent, retrieval-augmented system for analyzing
> small-angle scattering (SAS) data.

---

## 1. Goal

Today, `sas_llm/atomgpt_llm.py` uses a **single** LLM agent that generates an
MC-DFM Python script from a natural-language description. This design extends
that into a **team of specialized agents** that can:

1. Choose the best analysis tool for a given dataset (MC-DFM, SasView, or McSAS)
2. Run the chosen tool to fit the data
3. Judge the quality of the resulting fit and decide whether to retry or switch tools
4. Be augmented with retrieved context — past analyses, papers, tool docs
5. Accept user feedback mid-analysis and remember preferences across sessions

All of this is built on the `openai-agents` SDK (the `agents` package) already
used in the repo, talking to the AtomGPT API.

---

## 2. The Analysis Tools (the "hands")

Each tool is wrapped in a Python function with a **uniform interface** so the
agents can call any of them identically.

| Tool | Strength | Best for | Wrapper source |
|---|---|---|---|
| **MC-DFM** (this repo) | Real-space, atomistic, large hierarchical assemblies | Proteins, crystals, tubes, building blocks from PDB | `Scattering_Simulator/fitting.py` |
| **SasView / sasmodels** | ~100 analytical form + structure factors, fast | Standard shapes (spheres, cylinders, core-shell) | `sasmodels` Python API |
| **McSAS** | Monte Carlo size/shape **distribution** retrieval, form-free | Polydisperse systems needing a distribution, not a single fit | `mcsas` Python package |

### Uniform wrapper contract

Every tool exposes the same signature and returns the same result object:

```python
def run_<tool>(q, I, dI, model_hint: str, params: dict) -> FitResult
```

```python
# FitResult (the interface contract for the whole system)
{
    "fitted_curve": np.ndarray,   # model I(q) on the same q grid
    "parameters":   dict,         # fitted parameter values
    "chi_squared":  float,        # goodness-of-fit metric
    "residuals":    np.ndarray,   # (I_exp - I_model) / dI
    "metadata":     dict,         # tool name, model used, runtime, etc.
}
```

> **Design decision #1:** the exact `FitResult` schema is the load-bearing
> contract. Settle this before building anything else.

---

## 3. The Agents (the "brains")

Five specialized agents, coordinated by the SDK's **handoffs**.

### Agent A — Router / Tool Selector
- **Input:** experimental data features (q-range, curve shape) + sample description
- **Job:** decide which tool fits best
- **Output (`output_type`):** `{tool, reason, suggested_model}`
- **RAG:** retrieves past analyses of similar data + tool-selection rules
- **Hands off to:** Agent B

### Agent B — Executor / Analyst
- **Job:** run the chosen tool via `@function_tool` wrappers
- Picks arguments (model name, parameter bounds, q-limits)
- Captures the `FitResult`
- **RAG:** retrieves example parameter ranges and setup code for the chosen tool
- **Hands off to:** Agent C

### Agent C — Judge / Critic
- **Job:** evaluate fit quality, quantitatively and physically
- Considers: chi-squared, residual structure (systematic vs. random), parameter
  plausibility, goodness across q-ranges
- **Output (`output_type`):** `{verdict, critique, suggestions}` where
  `verdict ∈ {good, retry, switch_tool}`
- **RAG:** retrieves criteria for good SAS fits + typical values for the material class
- **Decision branch:**
  - `good` → hand off to Agent D
  - `retry` → hand back to Agent B with new bounds
  - `switch_tool` → hand back to Agent A

### Agent D — Reporter
- Summarizes the final fit, parameters, plots, and reasoning into a
  human-readable report, saved to a results folder (mirrors `sas_llm_results/`).

### Agent E — Feedback Interpreter (see Section 6)
- Translates user feedback into pipeline actions and memory updates.

---

## 4. The RAG Layer (the "library")

Shared augmentation feeding all agents. Built once, queried per-agent.

### Knowledge sources
- **Papers** — publications from the README + broader SAS literature (PDF → chunked text)
- **Past analyses** — existing `sas_llm_results/` folders (input + script + outcome)
- **Tool documentation** — SasView model docs, McSAS docs, MC-DFM instruction files
- **Worked examples** — curated good fits with parameters and the data they fit

### Pipeline
```
Documents → chunk → embed → vector store
                                  │
Query (per agent) → top-k retrieval → injected into agent instructions
```

### Implementation choices
- **Embeddings:** local sentence-transformer or an embedding API
- **Vector store:** FAISS (already in `environment.yml` as `faiss-cpu`) or Chroma
- **Retrieval as a tool:** expose `retrieve_context(query, source_filter)` as a
  `@function_tool` so agents pull context on demand (*agentic RAG*) rather than
  only stuffing it into the system prompt
- **Per-agent filtering:** same store, different `source_filter` — Router gets
  decision rules, Executor gets setup examples, Judge gets evaluation criteria

---

## 5. Core Data Flow

```
                    ┌─────────────────────────────┐
                    │   Vector store (FAISS)       │
                    │ papers · past fits · docs    │
                    └─────────────┬───────────────┘
                                  │ retrieve_context() tool
   experimental      ┌───────────┴───────────────────────────┐
   data (q,I,dI) ──► │                                         │
                     ▼                                         │
              ┌─────────────┐  handoff   ┌──────────────┐  handoff   ┌───────────┐
              │  A: Router  │ ─────────► │ B: Executor  │ ─────────► │ C: Judge  │
              │ pick tool   │            │ run tool via │            │ score fit │
              └─────────────┘            │ function_tool│            └─────┬─────┘
                     ▲                   └──────────────┘                  │
                     │  "switch_tool"          ▲   "retry" (new bounds)    │
                     └─────────────────────────┴───────────────────────────┘
                                                                           │ "good"
                                                                           ▼
                                                                    ┌───────────┐
                                                                    │ D: Report │
                                                                    └───────────┘
```

The whole thing runs inside **one `Runner.run()`** — handoffs and the agent loop
continue until the Judge says "good" or `max_turns` is reached.

---

## 6. User Feedback + Memory

### 6.1 Agent E — Feedback Interpreter

Sits between the user and the pipeline. The user can interject at any time.

| User says | Interpreted as | Routed to |
|---|---|---|
| "tighten the q-range to 0.02–0.3" | setup change | Agent B, re-run |
| "that radius is unphysical" | constraint change | Agent B with new bounds |
| "try a different tool" | tool change | Agent A |
| "the fit looks bad at low-q" | evaluation hint | Agent C |
| "always assume polydispersity for my samples" | **persistent preference** | Memory (Tier 2) |

- **Output (`output_type`):** `{action_type, target_agent, changes, persist: bool}`

### 6.2 Two tiers of memory

There are two *different kinds* of memory, with different storage and lifecycle.

#### Tier 1 — Session memory (short-term, conversational)
- **What:** the back-and-forth of *this* analysis — what's been tried, Judge
  comments, user requests
- **How:** `SQLiteSession` (built into the SDK)
- **Scope:** one analysis; enables "go back to the previous parameters"
- **Lifecycle:** cleared when starting a new dataset

#### Tier 2 — Preference / knowledge memory (long-term, persistent)
- **What:** durable guidance for *future* analyses too — e.g. "this user's
  samples are always polydisperse," "prefer McSAS for size distributions,"
  "constrain protein radii to 20–80 Å"
- **How:** hybrid store —
  - **hard constraints** (bounds, tool preferences) → structured rules file (JSON/SQLite), read into agent instructions every run
  - **soft guidance** (style, sample context) → vector-store entries with
    `source="user_preference"`, retrieved by relevance
- **Lifecycle:** persists across sessions; user can review/edit/delete

#### How the tiers feed the agents
```
Long-term preferences ──┐
                        ├──► injected into Router + Executor instructions every run
Session memory ─────────┘
```

### 6.3 Feedback + memory loop

```
                              ┌──────────────────────────────┐
                              │  Tier 2: Preference memory    │
   user feedback              │  (persistent, cross-session)  │
   "constrain R < 100Å    ┌──►│  structured rules + vectors   │
    and always assume      │   └───────────────┬──────────────┘
    polydispersity"        │                   │ "persist: true"
        │                  │                   │ read every run
        ▼                  │                   ▼
   ┌─────────────┐         │           ┌───────────────┐
   │ E: Feedback │ ────────┘           │ A/B/C agents  │
   │ Interpreter │ ─────────────────►  │ (the pipeline)│
   └─────────────┘   "persist: false"  └───────┬───────┘
        ▲             apply to this run only    │
        │                                        │ writes
        │            ┌───────────────────────────┘
        │            ▼
        │   ┌──────────────────────┐
        └───│ Tier 1: Session mem  │
            │ (SQLiteSession)      │
            └──────────────────────┘
```

> **Why two tiers:** if all feedback goes into one persistent memory, the agent
> starts applying one-off instructions ("for this dataset, fix the radius") to
> unrelated future analyses. Keeping transient (Tier 1) and durable (Tier 2)
> memory separate — and requiring the user to *explicitly* promote something to
> durable — prevents "over-remembering."

---

## 7. Mapping to the openai-agents SDK

| Need | SDK feature |
|---|---|
| Agents pass control A→B→C | **handoffs** |
| Agents run real tools | **`@function_tool`** wrappers around MC-DFM / SasView / McSAS |
| Structured decisions (tool choice, verdict, feedback action) | **`output_type`** (Pydantic models) |
| Retry/switch loop without infinite spin | Runner **agent loop** + **`max_turns`** |
| Short-term session memory | **`SQLiteSession`** |
| Pull papers/examples on demand | retrieval **`@function_tool`** (agentic RAG) |
| Reject non-physical requests / bad data | **guardrails** |
| Live UI updates | **`Runner.run_streamed()`** |
| Provider = AtomGPT | `AsyncOpenAI(base_url="https://atomgpt.org/api", ...)` |

---

## 8. Gradio UI Additions

Extend `sas_llm/gradio_app.py` with:
- **Data upload** — load experimental `(q, I, dI)` files
- **Chat box** — user types feedback, sees streamed agent responses
- **"Save as preference" toggle** — mark feedback persistent (Tier 2) vs. one-off (Tier 1)
- **Preferences panel** — view/edit/delete long-term preferences (memory must be
  transparent and correctable so users understand *why* the agent keeps making a choice)
- **Results view** — fitted curve, residuals, parameters, and the agent report

---

## 9. Build Phases

| Phase | Deliverable |
|---|---|
| **1. Tool wrappers** | Three `run_<tool>()` functions returning a common `FitResult`; unit-tested on sample data (no LLM yet) |
| **2. Single executor agent** | One agent + three `function_tools`; manual tool choice |
| **3. Multi-agent loop** | Add Router + Judge + handoffs with retry/switch logic |
| **4. RAG layer** | Vector store from papers + `sas_llm_results/`; `retrieve_context` tool |
| **5. Reporter + base UI** | Agent D + Gradio: upload → run (streamed) → fit + report |
| **6. Session memory + feedback** | `SQLiteSession` + Agent E; user steers one analysis |
| **7. Persistent preferences** | Tier-2 store; `persist` feedback shapes future runs |
| **8. UI polish** | Chat box, save-as-preference toggle, preferences panel |

> Build Phase 1 solidly first — every later phase depends on the uniform
> `FitResult` contract.

---

## 10. Open Design Decisions

Settle these before Phase 1:

1. **`FitResult` schema** — exact fields every tool must return
2. **Evaluation metric** — what chi-squared / residual criteria define "good" for
   the Judge, and whether it's consistent across tools that fit differently
3. **Retry budget** — how many retry/switch cycles before giving up (`max_turns`)
4. **Tool availability** — are `sasmodels` and `mcsas` pip-installable in this
   environment, or do they need subprocess calls? This shapes the wrapper design.
5. **Tier-2 preference schema** — structure of the persistent rules store

---

## 11. Relationship to Existing Code

| Existing | Role in new design |
|---|---|
| `sas_llm/atomgpt_llm.py` | Pattern for agent creation + AtomGPT client; evolves into the multi-agent orchestrator |
| `sas_llm/instructions_*.txt` | Become the base instructions for the Executor agent (MC-DFM tool) |
| `Scattering_Simulator/fitting.py` | Wrapped as the MC-DFM `function_tool` |
| `sas_llm_results/` | Ingested into the RAG vector store as past examples |
| `sas_llm/gradio_app.py` | Extended with data upload, chat, and preferences UI |
| `faiss-cpu` (in `environment.yml`) | Backs the RAG vector store |
```
