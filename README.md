# FinTox SmartBilling

> **Enterprise Financial Toxicity Reduction Platform for Oncology Patients**  
> Deterministic claim sequence optimization powered by a stacked assistance program engine, BM25 RAG knowledge retrieval, and a Gemini LLM narrative layer — built to align with the [Flatiron Health Smart Billing framework](https://resources.flatiron.com/flatiron-stories/part-1-financial-toxicity).

---

## Table of Contents

1. [Purpose](#purpose)
2. [Core Concept — Smart Billing vs Traditional Billing](#core-concept)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Data Layer](#data-layer)
6. [Engine Layer (src/)](#engine-layer)
   - [policy_registry.py](#policy_registrypy)
   - [policy_loader.py](#policy_loaderpy)
   - [risk_gate.py](#risk_gatepy)
   - [rag_engine.py](#rag_enginepy)
   - [orchestrator.py](#orchestratorpy)
7. [RAG Pipeline](#rag-pipeline)
8. [Stacking Calculation Flow](#stacking-calculation-flow)
9. [UI Layer (app.py)](#ui-layer)
10. [Installation](#installation)
11. [Running the Application](#running-the-application)
12. [Patient Profiles](#patient-profiles)
13. [Policy Catalog](#policy-catalog)
14. [Key Metrics Explained](#key-metrics-explained)
15. [Eligibility Rules Engine](#eligibility-rules-engine)
16. [Design Decisions](#design-decisions)
17. [Dependencies](#dependencies)

---

## Purpose

Financial toxicity — the economic burden of cancer treatment — is one of the leading causes of treatment non-adherence and bankruptcy among oncology patients. Manufacturer copay assistance cards and charitable foundation grants exist to offset this burden, but **the order in which claims are submitted to insurance determines who absorbs the patient's deductible**.

FinTox SmartBilling solves three problems:

| Problem | Solution |
|---|---|
| Billing staff submit claims in arbitrary order, leaving money on the table | Deterministic simulation of two billing paths — traditional (facility first) vs smart (drug first) |
| Copay cards and foundation grants are treated as competing programs | Stacked engine: card is primary (drug-only), grant is secondary (residual costs) |
| Clinic staff cannot quickly answer patient eligibility questions | BM25 RAG + Gemini LLM answer questions exclusively from policy documents |

---

## Core Concept

### Traditional Billing (Path A) — Facility Submitted First

```
Patient deductible: $500
Claim 1 → Facility fee ($1,500)
  Insurance applies $500 deductible → patient owes $500 ded + $200 coins = $700
  Manufacturer card CANNOT cover facility fees → patient absorbs $700
  Foundation grant covers $700 → grant budget consumed: $700

Claim 2 → Drug ($2,500) — deductible already exhausted
  Patient owes $0 ded + $500 coins = $500
  Manufacturer card covers $500 → card budget consumed: $500
  Foundation grant: $0 needed

Total patient OOP: $0    Foundation grant used: $700
```

### Smart Billing (Path B) — Drug Submitted First ✅

```
Patient deductible: $500
Claim 1 → Drug ($2,500) — submitted FIRST
  Insurance applies $500 deductible → patient owes $500 ded + $400 coins = $900
  Manufacturer card covers $900 (deductible + coinsurance) → card budget consumed: $900
  Foundation grant: $0 needed (card handled it)

Claim 2 → Facility fee ($1,500) — deductible now $0
  Patient owes $0 ded + $300 coins = $300
  Manufacturer card CANNOT cover facility → $300 remains
  Foundation grant covers $300 → grant budget consumed: $300

Total patient OOP: $0    Foundation grant used: $300
```

**Foundation Grant Preserved = $700 − $300 = $400**

Same patient outcome ($0 OOP), but smart billing saves $400 of the foundation's finite annual grant — preserving it for the patient's future treatment cycles.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         app.py  (Streamlit UI)                          │
│                                                                         │
│  ┌──────────────┐   ┌────────────────────────────────────────────────┐  │
│  │   Sidebar    │   │                 Main Panel                     │  │
│  │  (Form Input)│   │  ┌─────────────────┐  ┌──────────────────────┐│  │
│  │              │   │  │ Simulation Panel │  │  Chat Interface      ││  │
│  │ Patient data │   │  │ (Run Preview     │  │  (Q&A, bullet-point  ││  │
│  │ Drug code    │   │  │  button only)    │  │   LLM responses)     ││  │
│  │ Costs/       │   │  └────────┬─────────┘  └──────────┬───────────┘│  │
│  │ deductible   │   │           │                        │            │  │
│  └──────────────┘   └───────────┼────────────────────────┼────────────┘  │
└──────────────────────────────── │ ──────────────────────│ ─────────────┘
                                  │                        │
                    ┌─────────────▼────────────────────────▼──────────────┐
                    │           orchestrator.py                            │
                    │  1. Calls select_best_stack (risk_gate)              │
                    │  2. Enriches RAG query with winning policy files     │
                    │  3. Retrieves policy docs (rag_engine)               │
                    │  4. Builds LLM prompt (sim block + policy docs)      │
                    │  5. Calls Gemini LLM for bullet-point narrative      │
                    └────────────────┬─────────────────────┬──────────────┘
                                     │                     │
               ┌─────────────────────▼──────┐   ┌─────────▼──────────────┐
               │        risk_gate.py        │   │      rag_engine.py      │
               │                            │   │                         │
               │  _is_policy_eligible()     │   │  BM25 keyword index     │
               │  _apply_card()             │   │  over description:      │
               │  _apply_stacked()          │   │  prose blocks from      │
               │  calculate_stacked_        │   │  all 10 YAML files      │
               │    billing_paths()         │   │                         │
               │  select_best_stack()       │   │  Exact billing-code     │
               └─────────────┬──────────────┘   │  boost (J9331, etc.)   │
                             │                  └─────────────────────────┘
               ┌─────────────▼──────────────┐
               │       policy_loader.py     │
               │                            │
               │  COPAY_CARD_REGISTRY       │
               │    (5 manufacturer cards)  │
               │  ASSISTANCE_POLICY_REGISTRY│
               │    (5 foundation grants)   │
               └─────────────┬──────────────┘
                             │
               ┌─────────────▼──────────────┐
               │        data/               │
               │  copay_cards/*.yaml        │
               │  assistance_policies/*.yaml│
               │  patients.json             │
               └────────────────────────────┘
```

---

## Project Structure

```
FinTox-SmartBilling/
└── fintox_smartbilling_v3/
    ├── app.py                          # Streamlit UI — sidebar form + chat panel
    ├── requirements.txt
    ├── .env                            # GOOGLE_API_KEY (not committed)
    ├── .env.example
    │
    ├── src/
    │   ├── __init__.py
    │   ├── policy_registry.py          # PolicyConfig Pydantic model + FPL_2024 table
    │   ├── policy_loader.py            # Builds COPAY_CARD_REGISTRY + ASSISTANCE_POLICY_REGISTRY
    │   ├── risk_gate.py                # Stacked simulation engine (all arithmetic lives here)
    │   ├── rag_engine.py               # BM25 knowledge base over YAML prose descriptions
    │   └── orchestrator.py             # Pipeline controller: simulation → RAG → LLM
    │
    └── data/
        ├── patients.json               # 7 seed patient profiles
        ├── copay_cards/                # Manufacturer copay card YAML files
        │   ├── company_x_copay_card.yaml
        │   ├── abbvie_myabbvie_assist.yaml
        │   ├── astrazeneca_az_assist.yaml
        │   ├── genentech_access_card.yaml
        │   └── pfizer_oncology_together.yaml
        └── assistance_policies/        # Foundation / charitable grant YAML files
            ├── pan_foundation_breast.yaml
            ├── leukemia_lymphoma_society.yaml
            ├── cancer_care_copay_assist.yaml
            ├── healthwell_foundation.yaml
            └── patient_advocate_foundation.yaml
```

---

## Data Layer

### YAML Policy Schema

Every policy — whether a manufacturer copay card or a foundation grant — is a single YAML file with two sections:

**Structured fields** (parsed by `policy_loader.py` → `PolicyConfig`):

```yaml
policy_id: CX-COPAY-2024
file_name: company_x_copay_card.yaml
program_name: Company X Copay Assistance Card
program_type: manufacturer_card          # or foundation_grant

eligible_billing_codes: [J9331]          # empty list = any code
eligible_diagnoses: [breast, her2]       # OR-matched, word-boundary
required_biomarkers: []                  # AND-matched; empty = none required

requires_commercial_insurance: true      # rejects Medicare/Medicaid/TRICARE
requires_insurance: true

covers_facility_fees: false              # critical legal restriction
coverage_pct: 1.0                        # 100% of patient responsibility
monthly_max: null
annual_max: 25000.00
covers_deductible: true                  # false = coinsurance-only (e.g. PAF)

income_limit_fpl_pct: null               # family-size-adjusted FPL ceiling
income_limit_absolute: null              # flat dollar cap (e.g. $150,000)
requires_active_treatment: true
```

**Prose description block** (indexed by `rag_engine.py` for BM25 retrieval — never passed to simulation):

```yaml
description: |
  Full plain-English policy narrative including enrollment steps,
  required documentation, claim sequencing instructions, exclusions,
  and contact information...
```

The split is intentional: **structured fields drive math, prose drives answers**.

### patients.json

Each patient record contains:

```json
{
  "id": "PT-001",
  "name": "Maria",
  "diagnosis": "Stage II HER2+ Breast Cancer",
  "annual_household_income": 42000.00,
  "family_size": 3,
  "insurance_type": "commercial",
  "biomarkers": ["HER2+", "ER-", "PR-"],
  "treatment_status": "active",
  "insurance": {
    "remaining_deductible": 500.00,
    "coinsurance_rate": 0.20
  },
  "claims": [
    { "type": "drug",         "billing_code": "J9331", "retail_cost": 2500.00 },
    { "type": "facility_fee", "billing_code": "ADMIN-FEE", "retail_cost": 1500.00 }
  ]
}
```

---

## Engine Layer

### policy_registry.py

Defines the `PolicyConfig` Pydantic model — the single typed representation of any assistance program — and the FPL 2024 lookup table used for income-ceiling checks.

```python
class PolicyConfig(BaseModel):
    policy_id: str
    program_type: str              # "manufacturer_card" | "foundation_grant"
    eligible_billing_codes: list[str]
    eligible_diagnoses: list[str]
    required_biomarkers: list[str]
    covers_facility_fees: bool     # Manufacturer cards: always False
    coverage_pct: float
    annual_max: float
    covers_deductible: bool        # PAF-style grants: False (coinsurance only)
    income_limit_fpl_pct: Optional[float]
    income_limit_absolute: Optional[float]
    ...
```

**FPL 2024 table** (`FPL_2024`) maps family size (1–8) to federal poverty level dollar amounts. Income ceiling = `FPL_2024[family_size] × income_limit_fpl_pct / 100`.

---

### policy_loader.py

Reads all YAML files at startup and builds two separate registries:

```
data/copay_cards/*.yaml     ──►  COPAY_CARD_REGISTRY        (5 programs)
data/assistance_policies/*.yaml ──►  ASSISTANCE_POLICY_REGISTRY  (5 programs)

COPAY_CARD_REGISTRY + ASSISTANCE_POLICY_REGISTRY  ──►  POLICY_REGISTRY  (RAG only)
```

The registries are module-level constants — loaded once per server start. Adding or modifying a policy requires only editing the YAML file; no Python code changes needed.

---

### risk_gate.py

The deterministic simulation core. **No LLM involvement.** All arithmetic is Python; all outputs are Pydantic-validated.

#### Data Models

```
ClaimLineResult
  ├── retail_cost
  ├── deductible_applied
  ├── coinsurance_applied
  ├── copay_card_covers      ← manufacturer card contribution
  ├── foundation_covers      ← foundation grant contribution
  └── patient_oop            ← what remains (always >= 0)

BillingPathResult
  ├── claim_sequence: list[ClaimLineResult]
  ├── total_patient_oop
  ├── total_copay_card_absorbs
  └── total_foundation_absorbs

ComparativeBillingResult
  ├── primary_policy          ← winning copay card
  ├── secondary_policy        ← winning foundation grant
  ├── path_a_traditional      ← BillingPathResult (facility first)
  ├── path_b_smart            ← BillingPathResult (drug first)
  ├── net_savings_smart_vs_traditional
  └── foundation_grant_preserved   ← Path A grant − Path B grant
```

Pydantic `field_validator` enforces no-negative-value integrity on every financial field. A `model_validator` cross-checks that `net_savings` equals `path_a_oop − path_b_oop` and computes `foundation_grant_preserved` automatically.

#### Core Functions

**`_is_policy_eligible(patient, policy) → (bool, reason_str)`**

Runs 9 sequential eligibility gates:

```
1. Drug claim present?
2. Billing code in eligible_billing_codes?  (empty list = any code accepted)
3. Diagnosis contains at least one keyword? (word-boundary regex, OR-logic)
4. All required biomarkers present?         (AND-logic)
5. Commercial insurance required?           (rejects Medicare/Medicaid/TRICARE/VA)
6. Any insurance required?                  (rejects uninsured)
7. Active/initiating treatment status?
8. Income ≤ FPL-adjusted ceiling?
9. Income ≤ absolute dollar cap?
```

**`_apply_card(patient_gross, policy, card_spent_so_far, deductible_component) → float`**

Calculates the dollar amount one program covers on a single claim, respecting:
- `covers_deductible` flag (False = coinsurance-only programs skip deductible)
- `monthly_max` cap
- `annual_max` minus `card_spent_so_far`
- `coverage_pct` (most programs = 1.0 = 100%)

**`_apply_stacked(claim_gross, deductible_component, copay_card, foundation, card_spent, grant_spent, is_facility_claim) → (card_covers, foundation_covers, patient_oop)`**

Applies the two-layer stack to one claim:
1. Manufacturer card covers first (skips entirely if `is_facility_claim=True` and `covers_facility_fees=False`)
2. Foundation grant covers remaining patient balance (respects its own `covers_facility_fees` flag)

**`calculate_stacked_billing_paths(patient, copay_card, foundation) → ComparativeBillingResult`**

Runs both paths in full:

```
PATH A (Traditional — facility first):
  Claim 1: Facility fee  → _apply_stacked(..., is_facility=True)
  Claim 2: Drug          → _apply_stacked(..., is_facility=False)

PATH B (Smart Billing — drug first):
  Claim 1: Drug          → _apply_stacked(..., is_facility=False)
  Claim 2: Facility fee  → _apply_stacked(..., is_facility=True)
                           ↑ deductible already zero after drug
```

**`select_best_stack(patient, copay_cards, foundations) → (primary, secondary, result, card_rankings, grant_rankings)`**

Evaluates every eligible `(card, foundation)` combination and selects the best stack by:

```
Sort key:
  1. Path B patient OOP (ascending)     — minimize patient burden first
  2. Path B foundation grant spend      — preserve grant budget second
  3. Combined annual_max (descending)   — tiebreaker: more headroom
```

Returns separate ranking lists for cards and grants (used by the UI comparison table).

---

### rag_engine.py

BM25 keyword retrieval over the prose `description:` blocks from all 10 YAML policy files. ChromaDB and vector embeddings are intentionally avoided — BM25 is sufficient and faster for 10 documents with known billing-code query patterns.

#### How it Works

```
At startup:
  Load all *.yaml from copay_cards/ and assistance_policies/
  Extract description: prose → Document objects
  Extract eligible_billing_codes → stored as metadata (not indexed text)
  Build BM25 index over tokenized prose

At query time:
  Query → detect billing code (e.g. "J9331") via regex
  BM25 rank all 10 documents
  If billing code found: force-promote documents whose YAML lists that code
  Return top_k=3 documents with metadata
```

#### Billing Code Exact-Match Boosting

```
Normal BM25 ranking (by semantic relevance):
  1. PAN Foundation (mentions "breast cancer")
  2. HealthWell (mentions "breast")
  3. Company X (mentions "Trastuzumab")

After billing code boost (query contains "J9331"):
  1. Company X  ← forced first: eligible_billing_codes: [J9331]
  2. PAN Foundation
  3. HealthWell
```

This ensures the program that actually covers the patient's drug always surfaces in the context window, regardless of how the question is phrased.

---

### orchestrator.py

The pipeline controller. Called once per chat query.

```
run(patient, user_query):
  │
  ├─ 1. select_best_stack(patient, COPAY_CARD_REGISTRY, ASSISTANCE_POLICY_REGISTRY)
  │       → winning_primary, winning_secondary, simulation_result, card_rankings, grant_rankings
  │
  ├─ 2. Enrich RAG query with policy file names and billing codes
  │       enriched_query = f"{user_query} {primary.file_name} {secondary.file_name} {billing_code}"
  │
  ├─ 3. rag_engine.search(enriched_query)
  │       → top 3 policy description excerpts
  │
  ├─ 4. Build LLM prompt:
  │       [SIMULATION DATA]  ← pre-computed numbers (LLM must not alter)
  │       [POLICY DOCUMENTS] ← RAG-retrieved prose excerpts
  │       [USER QUESTION]    ← verbatim user query
  │
  └─ 5. Gemini LLM (temperature=0.0) → bullet-point narrative
```

**System prompt constraints** (strictly enforced):
- Respond ONLY in bullet points
- Every bullet sourced from `[SIMULATION DATA]` or `[POLICY DOCUMENTS]`
- No external knowledge
- No arithmetic (all numbers come from simulation block)
- 3–6 bullets maximum
- Cite source inline as `[Source: filename.yaml]`
- Yes/no questions: first bullet is `Yes — <reason>` or `No — <reason>`

---

## RAG Pipeline

```
User query: "What documents does Maria need to enroll?"
                         │
                         ▼
        ┌────────────────────────────────┐
        │  Billing code extraction       │
        │  regex: r"\b[A-Z][0-9A-Z]+\b" │
        │  → "J9331" found in query      │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  BM25 ranking (10 documents)   │
        │  Query tokens: DOCUMENTS MARIA │
        │  NEED ENROLL J9331             │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Billing code boost            │
        │  company_x_copay_card.yaml     │
        │  (has J9331) → rank 1          │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Top 3 excerpts returned:      │
        │  1. company_x_copay_card.yaml  │
        │     "...proof of commercial    │
        │     insurance, valid Rx from   │
        │     licensed oncologist,       │
        │     signed enrollment form..." │
        │  2. pan_foundation_breast.yaml │
        │  3. healthwell_foundation.yaml │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  LLM prompt assembly           │
        │  [SIMULATION DATA] + excerpts  │
        │  + user question               │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Gemini response (bullets):    │
        │  • Proof of commercial         │
        │    insurance required          │
        │    [Source: company_x...]      │
        │  • Valid oncologist Rx for     │
        │    J9331 [Source: company_x...]│
        │  • Signed patient enrollment   │
        │    form [Source: company_x...] │
        └────────────────────────────────┘
```

---

## Stacking Calculation Flow

```
select_best_stack(patient, copay_cards=[5], foundations=[5])
│
├─ Eligibility check loop (cards)
│   ├─ Company X   → eligible? ✅ (J9331 match, commercial, HER2+)
│   ├─ AbbVie      → eligible? ❌ (J8999-VEN not in patient claims)
│   ├─ AstraZeneca → eligible? ❌ (BRCA1 required, not in biomarkers)
│   ├─ Genentech   → eligible? ❌ (J9306 billing code mismatch)
│   └─ Pfizer      → eligible? ❌ (J8999 billing code mismatch)
│
├─ Eligibility check loop (grants)
│   ├─ PAN Foundation → eligible? ✅ (breast diagnosis, commercial)
│   ├─ HealthWell     → eligible? ❌ (diagnosis mismatch)
│   ├─ LLS            → eligible? ❌ (leukemia/lymphoma required)
│   ├─ Cancer Care    → eligible? ✅ (broad eligibility)
│   └─ PAF            → eligible? ✅ (broad eligibility)
│
├─ Build stack matrix:
│   eligible_cards  = [Company X]
│   eligible_grants = [PAN, Cancer Care, PAF]
│
│   → (Company X + PAN)         → calculate_stacked_billing_paths
│   → (Company X + Cancer Care) → calculate_stacked_billing_paths
│   → (Company X + PAF)         → calculate_stacked_billing_paths
│
├─ Rank by (Path B OOP ↑, grant spend ↑, annual_max ↓)
│   1. Company X + PAN  → OOP $0, grant $300  ← WINNER
│   2. Company X + PAF  → OOP $0, grant $300
│   3. Company X + CCAF → OOP $0, grant $350
│
└─ Return: primary=Company X, secondary=PAN, result=..., card_rankings, grant_rankings
```

---

## UI Layer

### Sidebar (Structured Input Form)

- Patient selector dropdown (loads seed data from `patients.json`)
- Manual override fields: name, diagnosis, insurance type, income, family size, biomarkers, deductible, coinsurance rate, drug billing code, drug cost, facility cost
- **▶ Run Simulation Preview** button — triggers simulation and renders the full comparison panel; clears chat history so the next query starts fresh

### Main Panel

**Simulation Panel** (appears only when "Run Simulation Preview" is clicked):

```
┌──────────────────────────────────────────────────────────────────┐
│ 📊 Comparative Billing Simulation                                │
│ Primary (Copay Card): Company X  |  Secondary: PAN Foundation    │
│                                                                  │
│  Path A OOP │  Path B OOP │  Patient Savings │  Grant Preserved  │
│  $700.00    │  $0.00      │  $700.00         │  $400.00          │
│                                                                  │
│  🏆 All Assistance Programs — Policy Comparison Table            │
│  ┌─────────── Manufacturer Copay Cards ──────────────┐           │
│  │ Program      │ Annual Max │ Ded │ Fac │ Selected  │           │
│  │ Company X    │ $25,000    │  ✅  │  —  │ ✅ PRIMARY│           │
│  └───────────────────────────────────────────────────┘           │
│  ┌─────────── Foundation Grants ─────────────────────┐           │
│  │ PAN Foundation │ $10,000 │  —   │  ✅  │ SECONDARY│           │
│  └───────────────────────────────────────────────────┘           │
│                                                                  │
│  📑 Detailed Claim-Line Breakdown  [Path A] [Path B]             │
└──────────────────────────────────────────────────────────────────┘
```

**Chat Panel** (no simulation in chat — answers only):

```
User: Is Maria eligible for LLS Financial Aid?

Assistant (bullet-point only, from data):
• No — Maria's diagnosis is "Stage II HER2+ Breast Cancer"; the LLS Patient
  Financial Aid program requires a leukemia or lymphoma diagnosis
  [Source: leukemia_lymphoma_society.yaml].
```

---

## Installation

### Prerequisites

- Python 3.10 or higher (tested on 3.14)
- A Google AI Studio API key ([get one here](https://aistudio.google.com/app/apikey))
- Git (optional)

### Step-by-Step

```
Clone or download the repository
            │
            ▼
┌───────────────────────────────────────────┐
│  cd FinTox-SmartBilling                   │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│  Create virtual environment               │
│                                           │
│  python -m venv venv                      │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│  Activate virtual environment             │
│                                           │
│  Windows PowerShell:                      │
│    venv\Scripts\Activate.ps1              │
│                                           │
│  Windows Git Bash / CMD:                  │
│    source venv/Scripts/activate           │
│                                           │
│  macOS / Linux:                           │
│    source venv/bin/activate               │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│  Install dependencies                     │
│                                           │
│  pip install -r                           │
│    fintox_smartbilling_v3/requirements.txt│
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│  Configure API key                        │
│                                           │
│  Copy fintox_smartbilling_v3/.env.example │
│  to fintox_smartbilling_v3/.env           │
│                                           │
│  Edit .env:                               │
│    GOOGLE_API_KEY=your_key_here           │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│  Run the application                      │
│                                           │
│  cd fintox_smartbilling_v3               │
│  streamlit run app.py                     │
└───────────────────┬───────────────────────┘
                    │
                    ▼
         Opens at http://localhost:8501
```

### .env File

```bash
# fintox_smartbilling_v3/.env
GOOGLE_API_KEY=AIza...your_key_here
```

---

## Running the Application

### Application Run Flow

```
Browser → localhost:8501
          │
          ▼
    Streamlit starts
    _load_patients()       ← reads patients.json (cached)
    _get_orchestrator()    ← builds BM25 index (cached per server start)
          │
          ▼
    Sidebar renders
    User selects patient or fills form manually
          │
          ▼
    ┌─────────────────────────────────────────────────────┐
    │ OPTION A: Click "▶ Run Simulation Preview"          │
    │                                                     │
    │  select_best_stack() runs immediately               │
    │  Simulation panel renders above chat               │
    │  Chat history is cleared                           │
    │  User can now ask questions about the result        │
    └──────────────────────────┬──────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────┐
    │ OPTION B: Type a question in chat directly          │
    │                                                     │
    │  orchestrator.run() called                         │
    │  select_best_stack() runs (inside orchestrator)    │
    │  RAG retrieves policy docs                         │
    │  Gemini LLM generates bullet-point response        │
    │  Chat shows: user message + LLM bullets only       │
    │  Audit trail expander shows source excerpts        │
    └─────────────────────────────────────────────────────┘
```

### Typical Workflow

1. **Select a patient** from the sidebar dropdown (or manually enter values)
2. **Click "▶ Run Simulation Preview"** to see the full side-by-side comparison
3. **Read the results** — 4 metrics, policy comparison tables, per-claim breakdown
4. **Ask questions in chat** — the AI answers in bullet points from policy documents

### Example Questions to Ask

```
• What documents does [patient] need to enroll in the copay card?
• Is [patient] eligible for [program name]?
• Why was [program] not recommended?
• What happens if [patient] switches to Medicare mid-year?
• What is the annual benefit limit for the selected copay card?
• Does the selected foundation grant cover the facility fee?
```

---

## Patient Profiles

| ID | Name | Diagnosis | Drug Code | Insurance | Income | Copay Card | Foundation |
|----|------|-----------|-----------|-----------|--------|------------|------------|
| PT-001 | Maria | HER2+ Breast Cancer | J9331 | Commercial | $42k/fam-3 | Company X | PAN Foundation |
| PT-002 | James | NSCLC Stage III | J9271 | Commercial | $80k/fam-1 | None eligible | PAF |
| PT-003 | Linda | BRCA1+ Ovarian Cancer | J8999-OLA | Commercial | $55k/fam-2 | AstraZeneca | HealthWell |
| PT-004 | Robert | HER2+ Metastatic Breast | J9306 | Commercial | $95k/fam-1 | Genentech | PAF |
| PT-005 | Marcus | CLL Relapsed/Refractory | J8999-VEN | Commercial | $38k/fam-3 | AbbVie | HealthWell |
| PT-006 | Dorothy | Colorectal Adenocarcinoma | J9035 | **Medicare** | $28k/fam-1 | None (Medicare) | HealthWell |
| PT-007 | Elena | TNBC Stage I | J8999 | Commercial | $120k/fam-2 | Pfizer\* | PAF |

\* Elena's income ($120k) is below Pfizer's $150,000 absolute cap but above many FPL-based program limits.

**Key scenarios illustrated:**
- **Maria** — Classic Flatiron stacking case: $400 foundation grant preserved
- **James** — No manufacturer card available (J9271 not covered by any card); grant-only path
- **Linda** — Biomarker gate: AstraZeneca requires BRCA1 documentation
- **Robert** — Negative grant preservation (−$240): smart billing invests grant to achieve $0 OOP
- **Dorothy** — Medicare patient: all manufacturer cards ineligible (federal anti-kickback statute)
- **Elena** — Income eligibility boundary: Pfizer's flat $150k cap tested

---

## Policy Catalog

### Manufacturer Copay Cards (`data/copay_cards/`)

| Program | Policy ID | Drug | Billing Code | Annual Max | Covers Deductible | Covers Facility |
|---------|-----------|------|--------------|------------|-------------------|-----------------|
| Company X Copay Assistance Card | CX-COPAY-2024 | Trastuzumab (Herceptin) | J9331 | $25,000 | ✅ | ❌ |
| AbbVie myAbbVie Assist | ABBVIE-MABA-2024 | Venetoclax (Venclexta) | J8999-VEN | $20,000 | ✅ | ❌ |
| AstraZeneca AZ&ME Assist | AZ-ASSIST-2024 | Olaparib (Lynparza) | J8999-OLA | $30,000 | ✅ | ❌ |
| Genentech Oncology Access Card | GNEN-OAC-2024 | Pertuzumab (Perjeta) | J9306 | $35,000 | ✅ | ❌ |
| Pfizer Oncology Together | PFZ-OT-2024 | Ribociclib (Kisqali) | J8999 | $18,000 | ✅ | ❌ |

> All manufacturer cards require **commercial insurance** (Medicare/Medicaid/TRICARE/VA ineligible under federal anti-kickback statute). None cover facility fees.

### Foundation Grants (`data/assistance_policies/`)

| Program | Policy ID | Eligible Diagnoses | Annual Max | Covers Deductible | Covers Facility | Income Limit |
|---------|-----------|-------------------|------------|-------------------|-----------------|--------------|
| PAN Foundation Breast Cancer | PAN-BREAST-2024 | Breast cancer | $10,000 | ❌ (coinsurance only) | ✅ | 500% FPL |
| Leukemia & Lymphoma Society | LLS-PFA-2024 | Leukemia, lymphoma, myeloma | $12,000 | ✅ | ✅ | 400% FPL |
| Cancer Care Copay Assist | CCAF-ONC-2024 | Any cancer | $8,000 | ❌ | ✅ | 400% FPL |
| HealthWell Foundation | HWF-ONC-2024 | Breast, ovarian, colorectal, CLL | $15,000 | ✅ | ✅ | 600% FPL |
| Patient Advocate Foundation | PAF-CRF-2024 | Any active cancer | $6,000 | ❌ (coinsurance only) | ✅ | None |

---

## Key Metrics Explained

### Path A — Traditional OOP
Patient out-of-pocket total when the **facility fee is submitted first**. The deductible is consumed by the facility claim, which the copay card cannot cover. The foundation grant must absorb more of the residual balance.

### Path B — Smart Billing OOP
Patient out-of-pocket total when the **drug claim is submitted first**. The copay card absorbs the deductible on the drug, leaving the facility claim with $0 deductible — only coinsurance remains, which the foundation grant covers at a lower cost.

### Patient OOP Savings
`Path A OOP − Path B OOP`

When both paths produce $0 OOP (which is common with a strong copay card), savings = $0. The real benefit in this case is Foundation Grant Preserved.

### Foundation Grant Preserved
`Path A foundation spend − Path B foundation spend`

- **Positive** — Smart billing uses less of the foundation's annual budget. More grant remains for future visits.
- **Negative** — Smart billing invests more grant to eliminate patient OOP. Still the right outcome: the grant absorbs cost so the patient pays nothing.
- **Zero** — Claim order doesn't affect grant usage (e.g., Medicare patients with no copay card).

---

## Eligibility Rules Engine

```
_is_policy_eligible(patient, policy)
│
├─ Gate 1: Drug claim present?
│     Fail → "No drug claim found"
│
├─ Gate 2: Billing code match?
│     policy.eligible_billing_codes = [J9331]
│     patient drug code = J9271
│     Fail → "J9271 not covered. This program covers: J9331"
│
├─ Gate 3: Diagnosis keyword match?
│     Uses word-boundary regex: r'\b' + keyword + r'\b'
│     (Prevents "all" in "LLS" matching "small" in "non-small cell")
│     Fail → "Diagnosis does not match any eligible condition"
│
├─ Gate 4: Biomarker check (AND-logic)?
│     policy.required_biomarkers = [BRCA1]
│     patient.biomarkers = [HER2+, ER-, PR-]
│     Fail → "Required biomarker(s) not documented: BRCA1"
│
├─ Gate 5: Commercial insurance required?
│     patient.insurance_type = "medicare"
│     Fail → "Manufacturer program requires commercial insurance"
│
├─ Gate 6: Any insurance required?
│     patient.insurance_type = "uninsured"
│     Fail → "Patient must have active insurance coverage"
│
├─ Gate 7: Active treatment required?
│     patient.treatment_status = "completed"
│     Fail → "Treatment status 'completed' does not meet active requirement"
│
├─ Gate 8: FPL income ceiling?
│     ceiling = FPL_2024[family_size] × income_limit_fpl_pct / 100
│     patient income > ceiling
│     Fail → "Income $80,000 exceeds 400% FPL ceiling of $60,240 for family of 1"
│
└─ Gate 9: Absolute income cap?
      policy.income_limit_absolute = 150000.0
      patient income > 150000.0
      Fail → "Income exceeds program's absolute cap of $150,000"

All gates pass → (True, "")
```

---

## Design Decisions

### Why BM25 instead of vector embeddings?

10 YAML documents with known billing code query patterns don't benefit from semantic similarity search. BM25 is deterministic, requires no GPU, needs no embedding API key, and correctly handles the exact-match priority (J9331 in query → J9331 document ranks first) that a cosine-similarity search would soften.

### Why separate copay card and foundation grant registries?

They are legally and mechanically distinct:
- Manufacturer cards are federally restricted to commercial insurance patients (anti-kickback statute)
- Cards target specific drugs by billing code; grants are diagnosis-based
- Cards cannot cover facility fees; grants typically can
- The stacking engine needs to evaluate them independently before pairing them

A single flat pool would require constant `program_type` filtering and is more error-prone.

### Why is the LLM prohibited from arithmetic?

LLMs hallucinate numbers. Dollar values in simulation output must match exactly what Python computed. The strict system prompt and `[SIMULATION DATA — DO NOT ALTER THESE NUMBERS]` block enforce this boundary. The LLM's only job is to surface relevant facts from the data blocks.

### Why does the simulation not appear in chat responses?

The simulation panel (with its 4 metrics, policy tables, and claim breakdown) is dense and is meant for structured review — not repeated in-line with a Q&A conversation. The "Run Simulation Preview" button provides the simulation in its proper context; chat responses are fast, focused, bullet-point answers.

### Why word-boundary regex for diagnosis matching?

Without `\b`, the keyword `"all"` (from LLS eligibility for leukemia/lymphoma) would match inside `"non-small cell lung cancer"`, falsely qualifying a lung cancer patient for an LLS leukemia grant. `re.search(r'\ball\b', ...)` correctly rejects it.

---

## Dependencies

```
streamlit==1.58.0              # UI framework
langchain==0.3.25              # LLM orchestration abstractions
langchain-community==0.3.24    # Community integrations
langchain-google-genai==2.1.4  # Gemini API integration
pydantic==2.13.4               # Runtime type validation for all financial models
bm25s==0.2.12                  # BM25 retrieval (Python 3.14 compatible)
python-dotenv==1.2.2           # .env file loading
pyyaml==6.0.2                  # YAML policy file parsing
```

**LLM:** `gemini-3.1-flash-lite` at `temperature=0.0` — deterministic, fast, low-cost.  
**No database required** — all state lives in YAML files and `patients.json`.  
**No vector store required** — BM25 runs entirely in-process.

---

## Adding a New Policy

1. Create a new YAML file in `data/copay_cards/` (manufacturer card) or `data/assistance_policies/` (foundation grant)
2. Fill in all structured fields following the schema in [Data Layer](#data-layer)
3. Write a thorough `description:` prose block (this is what the RAG engine indexes)
4. Restart the Streamlit server — the new policy is auto-loaded at startup

No Python code changes required.

## Adding a New Patient

Add a new entry to `data/patients.json` following the existing schema. The patient selector in the sidebar auto-populates from this file (cached at server start — restart required to see new entries).

---

*Built on the Flatiron Health Smart Billing framework for oncology financial toxicity reduction.*