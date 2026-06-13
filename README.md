# FinTox SmartBilling

An enterprise-grade oncology billing simulation platform that determines the optimal financial assistance stack for cancer patients and narrates the findings through a conversational AI assistant.

---

## Overview

Oncology patients face two types of out-of-pocket costs: drug infusion costs and facility/administrative fees. Two classes of financial assistance exist:

- **Manufacturer Copay Cards** — issued by drug makers. Legally restricted to commercial insurance patients; cannot cover facility fees.
- **Foundation Grants** — issued by independent nonprofits. Income-tested; accept government insurance (Medicare, Medicaid, TRICARE).

FinTox SmartBilling finds the optimal card + grant stack for a given patient, computes two billing paths (Traditional vs. Smart Billing), and surfaces OOP savings and preserved foundation grant. A Gemini-powered chat assistant answers follow-up questions grounded strictly in the simulation output and policy documents.

---

## Architecture

```
fintox_smartbilling_v3/
├── app.py                        # Streamlit UI — sidebar form, simulation display, chat
├── src/
│   ├── policy_registry.py        # PolicyConfig Pydantic model + FPL constants
│   ├── policy_loader.py          # Loads copay card + grant YAML files into registries
│   ├── risk_gate.py              # Deterministic eligibility checks + claim simulation engine
│   ├── rag_engine.py             # BM25 knowledge base over policy description prose
│   └── orchestrator.py          # Bridges simulation engine, RAG retrieval, and LLM
├── data/
│   ├── copay_cards/              # Manufacturer copay card policy YAML files
│   └── assistance_policies/      # Foundation grant policy YAML files
├── requirements.txt
└── .env.example
```

---

## Key Concepts

### Stacking Model
1. **Primary:** Manufacturer copay card — covers drug deductible and coinsurance. Cannot legally cover facility fees.
2. **Secondary:** Foundation grant — covers remaining patient balance.
3. **Selection:** The stack that minimises patient OOP and preserves the most foundation grant is chosen automatically.

---

## Billing Paths — Worked Example

**Scenario:** Drug infusion $2,500 · Facility fee $1,500 · Remaining deductible $500 · Coinsurance 20%
**Programs:** Copay card (covers drug deductible + coinsurance, cannot cover facility) · Foundation grant (covers remaining balance)

---

### Path A — Traditional Billing
Facility fee submitted **first**, drug claim second.

**Claim 1 — Facility fee ($1,500)**

| | |
|---|---|
| Deductible applied | $500 — full remaining deductible consumed here |
| Coinsurance (20% of $1,000 remaining) | $200 |
| Patient gross responsibility | $700 |
| Copay card covers | $0 — card cannot legally cover facility fees |
| Foundation grant covers | $700 |
| **Patient OOP** | **$0** |

**Claim 2 — Drug infusion ($2,500)**

| | |
|---|---|
| Deductible applied | $0 — already exhausted by facility claim |
| Coinsurance (20% of $2,500) | $500 |
| Patient gross responsibility | $500 |
| Copay card covers | $500 |
| Foundation grant covers | $0 |
| **Patient OOP** | **$0** |

**Path A totals — Patient OOP: $0 · Foundation grant consumed: $700 · Copay card consumed: $500**

---

### Path B — Smart Billing
Drug claim submitted **first**, facility fee second.

**Claim 1 — Drug infusion ($2,500)**

| | |
|---|---|
| Deductible applied | $500 — drug claim hits the deductible first |
| Coinsurance (20% of $2,000 remaining) | $400 |
| Patient gross responsibility | $900 |
| Copay card covers | $900 — card covers full drug deductible + coinsurance |
| Foundation grant covers | $0 |
| **Patient OOP** | **$0** |

**Claim 2 — Facility fee ($1,500)**

| | |
|---|---|
| Deductible applied | $0 — already exhausted by drug claim |
| Coinsurance (20% of $1,500) | $300 |
| Patient gross responsibility | $300 |
| Copay card covers | $0 — card cannot legally cover facility fees |
| Foundation grant covers | $300 |
| **Patient OOP** | **$0** |

**Path B totals — Patient OOP: $0 · Foundation grant consumed: $300 · Copay card consumed: $900**

---

### Why it matters

Both paths deliver $0 patient OOP, but the billing order changes everything:

| | Path A | Path B |
|---|---|---|
| Patient OOP | $0 | $0 |
| Foundation grant consumed | $700 | $300 |
| **Foundation grant preserved** | — | **$400** |

In Path A the facility fee arrives first and consumes the $500 deductible. The copay card is locked out of facility claims, so the foundation must absorb the full $700. By the time the drug arrives, the deductible is gone and the card only offsets $500 in coinsurance.

In Path B the drug arrives first. The copay card legally covers the drug's full $900 responsibility (deductible + coinsurance). The deductible is now exhausted, so the facility fee carries only $300 in coinsurance — the foundation covers $300 instead of $700.

The **$400 in preserved foundation grant** rolls forward to the next treatment visit. Across a monthly infusion schedule this compounds materially over a benefit year.

---

## Simulation Engine

`risk_gate.py` is entirely LLM-free. All arithmetic is deterministic and Pydantic-validated. Eligibility is checked across:

| Gate | Description |
|---|---|
| Billing code | Drug HCPCS/CPT code must match the program's covered codes |
| Insurance type | Commercial-only cards reject Medicare/Medicaid/TRICARE |
| Diagnosis | Programs may restrict to specific cancer types |
| Biomarkers | Some programs require confirmed mutation markers |
| Income (FPL) | Foundation grants apply family-size-adjusted FPL ceilings |
| Income (flat) | Some programs apply an absolute income cap |
| Treatment status | Most programs require active or initiating treatment |

---

## Policy Files

Each policy is a YAML file in `data/copay_cards/` or `data/assistance_policies/` with:

- **Structured fields** — parsed for deterministic eligibility checks (billing codes, annual max, coverage %, income limits, insurance requirements, etc.)
- **`description:` block** — free-prose description indexed by the RAG engine for LLM context

To add a new program, create a YAML file following the existing schema and restart the app. No code changes required.

### Current programs

| Type | Program |
|---|---|
| Copay Card | myAbbVie Assist Oncology Copay Card |
| Copay Card | AstraZeneca AZ&Me Assist Oncology Card |
| Copay Card | Genentech Oncology Access Card |
| Copay Card | Pfizer Oncology Together Copay Card |
| Copay Card | Company X Copay Assistance Card |
| Foundation Grant | Leukemia & Lymphoma Society (LLS) Patient Financial Aid |
| Foundation Grant | CancerCare Copay Assistance Foundation |
| Foundation Grant | HealthWell Foundation Oncology Grant |
| Foundation Grant | PAN Foundation Breast Cancer Grant |
| Foundation Grant | Patient Advocate Foundation (PAF) Co-Pay Relief |

---

## RAG — Retrieval-Augmented Generation

### What is RAG?

A standard LLM answers from training data alone — it cannot know the specific rules of a proprietary policy file you wrote yesterday. RAG solves this by retrieving relevant documents at query time and injecting them into the LLM prompt as grounding context. The model then answers from those documents rather than from memory, making responses accurate, citable, and auditable.

```
User question
     │
     ▼
Retrieval — search the document corpus for relevant excerpts
     │
     ▼
Augment — inject retrieved excerpts into the LLM prompt
     │
     ▼
Generate — LLM responds strictly from the provided excerpts
```

### How FinTox SmartBilling uses RAG

The policy corpus is small (10 YAML files) but the rules are highly specific — billing codes, FPL ceilings, biomarker requirements. A general-purpose LLM would hallucinate these details. RAG ensures every answer is grounded in the actual policy text.

**Document corpus:** The `description:` prose block from each YAML policy file is indexed — this is the human-readable explanation of what a program covers, who qualifies, and what exclusions apply. Structured eligibility fields (billing codes, annual max, income limits) are handled by the deterministic simulation engine, not the LLM.

**Retrieval method:** BM25 keyword search (via `bm25s`). Vector embeddings and ChromaDB are intentionally omitted — BM25 is sufficient for 10 documents and eliminates cold-start latency, embedding API costs, and infrastructure dependencies.

### RAG flow per query

```
User sends a chat message
        │
        ▼
1. SIMULATE (deterministic, no LLM)
   risk_gate runs eligibility checks + Path A / Path B arithmetic
   → produces structured simulation result with ALL programs: eligible + ineligible with reasons
        │
        ▼
2. RETRIEVE (guaranteed slots, no BM25 on user query)
   Slot 1 — selected primary copay card (always)
   Slot 2 — selected secondary foundation grant (always)
   Slot 3 — 2nd-best eligible copay card (if one exists)
   Slot 4 — 2nd-best eligible foundation grant (if one exists)
   Policy prose is fetched only for these guaranteed high-value documents
        │
        ▼
3. AUGMENT (prompt assembly)
   [SIMULATION DATA] block  — all programs listed with eligibility status + ineligibility reasons
   [POLICY DOCUMENTS] block — retrieved prose annotated with eligibility status inline
   Full prior conversation history prepended as HumanMessage / AIMessage turns
        │
        ▼
4. GENERATE (Gemini, temperature=0)
   Responds in bullet points
   Cites [Source: filename.yaml] for every policy rule referenced
   Prohibited from performing arithmetic — all numbers come from [SIMULATION DATA]
        │
        ▼
Chat response + knowledge base audit trail
```

### Why BM25 over vector search?

| | BM25 (used here) | Vector / semantic search |
|---|---|---|
| Setup | Zero — in-process, no external service | ChromaDB or Pinecone + embedding API |
| Cold start | None | Index build + embedding calls |
| Billing code recall | Exact token match → guaranteed | May miss exact codes |
| Corpus size | Optimal at < ~1,000 docs | Advantage grows at scale |
| Cost | Free | Embedding API per document + query |

For a fixed 10-document corpus of structured policy files, BM25 exact-keyword matching outperforms semantic search on the queries that matter most (billing codes, program names, diagnosis keywords).

---

## Setup

### Prerequisites

- Python 3.11+
- A Google AI Studio API key

### Installation

```bash
git clone https://github.com/avi267/FinTox-SmartBilling.git
cd FinTox-SmartBilling/fintox_smartbilling_v3

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
```

Edit `.env`:

```
GOOGLE_API_KEY=your-google-api-key-here
```

Get a key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. **Configure patient** — fill in the sidebar form with the patient's financial and clinical parameters.
2. **Run Simulation Preview** — click the button to see the Path A vs. Path B comparison and the policy eligibility table.
3. **Ask questions** — type in the chat panel. Examples:
   - *"Which program gives the best savings?"*
   - *"Why was the HealthWell grant disqualified?"*
   - *"What other saving opportunities are available?"*

The assistant responds in bullet points with inline source citations. It never performs arithmetic — all numbers come from the pre-computed simulation.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `langchain` + `langchain-google-genai` | LLM integration (Gemini) |
| `bm25s` | BM25 keyword retrieval for policy knowledge base |
| `pydantic` | Strongly-typed simulation result validation |
| `pyyaml` | Policy YAML file parsing |
| `python-dotenv` | Environment variable loading |
