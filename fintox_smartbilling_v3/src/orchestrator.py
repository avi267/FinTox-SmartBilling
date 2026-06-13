"""
orchestrator.py — Isolated Processing & Multi-Source Routing Controller
Bridges the UI, the risk_gate simulation engine, and the hybrid RAG knowledge base.
The LLM (gemini-3.1-flash-lite, temperature=0.0) is strictly prohibited from performing
arithmetic; it receives pre-computed values from risk_gate.py and must present them verbatim.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .risk_gate import select_best_stack, ComparativeBillingResult
from .policy_registry import PolicyConfig
from .policy_loader import COPAY_CARD_REGISTRY, ASSISTANCE_POLICY_REGISTRY
from .rag_engine import HighPrecisionKnowledgeBase


# ---------------------------------------------------------------------------
# Structured response data class
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResponse:
    user_query: str
    simulation_result: ComparativeBillingResult
    llm_narrative: str
    applied_policy: PolicyConfig        # primary card (backwards compat)
    primary_policy: PolicyConfig        # manufacturer card
    secondary_policy: PolicyConfig      # foundation grant
    all_card_rankings: list[dict]
    all_grant_rankings: list[dict]
    source_documents: list[dict] = field(default_factory=list)
    ineligible_documents: list[dict] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are FinTox-SmartBilling, an oncology billing engine assistant.

STRICT RESPONSE FORMAT — NO EXCEPTIONS:
- Respond ONLY in bullet points (markdown - list items).
- Every bullet must come directly from [SIMULATION DATA] or [POLICY DOCUMENTS]. \
No external knowledge, no general statements, no inferences beyond what is in those blocks.
- Answer ONLY what the question asks. No preamble, no intro sentence, no closing remarks.
- Do NOT repeat numbers the user can already see in the simulation panel; \
reference a figure only when it directly answers the question.
- NEVER perform arithmetic. All numbers come exclusively from [SIMULATION DATA].
- Cite policy source inline as [Source: filename.yaml] when referencing a rule or limit.
- 3–6 bullets maximum. Each bullet: one fact, one sentence.
- If a yes/no question: first bullet is "Yes — <reason>" or "No — <reason>".
- If the answer is not in the provided data blocks, say so in one bullet and stop.
"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_simulation_block(
    result: ComparativeBillingResult,
    card_rankings: list[dict] | None = None,
    grant_rankings: list[dict] | None = None,
) -> str:
    primary = result.primary_policy
    secondary = result.secondary_policy

    lines = [
        "[SIMULATION DATA — DO NOT ALTER THESE NUMBERS]",
        f"Patient: {result.patient_name}",
        f"Diagnosis: {result.diagnosis}",
        f"Remaining Deductible at Start: ${result.remaining_deductible_at_start:,.2f}",
        f"Coinsurance Rate: {result.coinsurance_rate * 100:.0f}%",
        "",
        "=== SELECTED PROGRAM STACK ===",
        f"Primary Program (Manufacturer Copay Card): {primary.program_name} (ID: {primary.policy_id})",
        f"Secondary Program (Foundation Grant):      {secondary.program_name} (ID: {secondary.policy_id})",
    ]

    if primary.policy_id == "NO-POLICY" and secondary.policy_id == "NO-POLICY":
        lines += [
            "⚠️ NO ELIGIBLE PROGRAM: No assistance program met all eligibility criteria for this patient.",
            "DO NOT recommend any financial assistance program. Explain that the patient is currently ineligible",
            "for all programs in the database and describe the disqualifying reason(s) if provided.",
        ]
    else:
        if primary.policy_id != "NO-POLICY":
            lines += [
                f"  Card Annual Max: ${primary.annual_max:,.2f}",
                f"  Card Covers Facility Fees: {'Yes' if primary.covers_facility_fees else 'No'}",
                f"  Card Covers Deductible: {'Yes' if primary.covers_deductible else 'No'}",
            ]
        if secondary.policy_id != "NO-POLICY":
            lines += [
                f"  Foundation Annual Max: ${secondary.annual_max:,.2f}",
                f"  Foundation Covers Facility Fees: {'Yes' if secondary.covers_facility_fees else 'No'}",
                f"  Foundation Covers Deductible: {'Yes' if secondary.covers_deductible else 'No'}",
            ]

    lines += [
        "",
        "=== PATH A: Traditional Billing ===",
        f"Description: {result.path_a_traditional.path_description}",
    ]

    for i, claim in enumerate(result.path_a_traditional.claim_sequence, start=1):
        lines += [
            f"  Claim {i}: {claim.claim_type} (Code: {claim.billing_code})",
            f"    Retail Cost:             ${claim.retail_cost:>10,.2f}",
            f"    Deductible Applied:      ${claim.deductible_applied:>10,.2f}",
            f"    Coinsurance:             ${claim.coinsurance_applied:>10,.2f}",
            f"    Copay Card Covers:       ${claim.copay_card_covers:>10,.2f}",
            f"    Foundation Grant Covers: ${claim.foundation_covers:>10,.2f}",
            f"    Patient OOP:             ${claim.patient_oop:>10,.2f}",
        ]

    lines += [
        f"  TOTAL Patient OOP (Path A):          ${result.path_a_traditional.total_patient_oop:>10,.2f}",
        f"  Total Copay Card Absorbs:            ${result.path_a_traditional.total_copay_card_absorbs:>10,.2f}",
        f"  Total Foundation Grant Absorbs:      ${result.path_a_traditional.total_foundation_absorbs:>10,.2f}",
        "",
        "=== PATH B: Smart Billing (Optimized) ===",
        f"Description: {result.path_b_smart.path_description}",
    ]

    for i, claim in enumerate(result.path_b_smart.claim_sequence, start=1):
        lines += [
            f"  Claim {i}: {claim.claim_type} (Code: {claim.billing_code})",
            f"    Retail Cost:             ${claim.retail_cost:>10,.2f}",
            f"    Deductible Applied:      ${claim.deductible_applied:>10,.2f}",
            f"    Coinsurance:             ${claim.coinsurance_applied:>10,.2f}",
            f"    Copay Card Covers:       ${claim.copay_card_covers:>10,.2f}",
            f"    Foundation Grant Covers: ${claim.foundation_covers:>10,.2f}",
            f"    Patient OOP:             ${claim.patient_oop:>10,.2f}",
        ]

    lines += [
        f"  TOTAL Patient OOP (Path B):          ${result.path_b_smart.total_patient_oop:>10,.2f}",
        f"  Total Copay Card Absorbs:            ${result.path_b_smart.total_copay_card_absorbs:>10,.2f}",
        f"  Total Foundation Grant Absorbs:      ${result.path_b_smart.total_foundation_absorbs:>10,.2f}",
        "",
        f"NET PATIENT SAVINGS (Smart vs Traditional): ${result.net_savings_smart_vs_traditional:>10,.2f}",
        f"FOUNDATION GRANT PRESERVED (Smart vs Traditional): ${result.foundation_grant_preserved:>10,.2f}",
        "  (Smart billing uses less of the foundation's annual grant, preserving it for future visits.)",
    ]

    # All programs with eligibility status — LLM uses this to answer any eligibility question
    all_rankings = list(card_rankings or []) + list(grant_rankings or [])
    lines += ["", "=== ALL PROGRAMS — ELIGIBILITY SUMMARY ==="]
    for r in all_rankings:
        p = r["policy"]
        if r["eligible"]:
            if p.policy_id == primary.policy_id:
                tag = "[ELIGIBLE — PRIMARY]  "
            elif p.policy_id == secondary.policy_id:
                tag = "[ELIGIBLE — SECONDARY]"
            else:
                tag = "[ELIGIBLE — OTHER]    "
            lines.append(
                f"  {tag} {p.program_name} | Annual Max: ${p.annual_max:,.0f} | "
                f"Covers Deductible: {'Yes' if p.covers_deductible else 'No'} | "
                f"Covers Facility: {'Yes' if p.covers_facility_fees else 'No'}"
            )
        else:
            reason = r.get("ineligibility_reason", "reason not specified")
            lines.append(f"  [INELIGIBLE]           {p.program_name} — {reason}")

    lines.append("[END SIMULATION DATA]")
    return "\n".join(lines)


def _format_policy_block(
    source_docs: list,
    all_rankings: list[dict] | None = None,
    primary_id: str = "",
    secondary_id: str = "",
) -> str:
    if not source_docs:
        return "[POLICY DOCUMENTS]\nNo policy documents retrieved.\n[END POLICY DOCUMENTS]"

    # Build eligibility lookup keyed by file_name
    eligibility: dict[str, tuple[bool, str]] = {}
    for r in (all_rankings or []):
        fname = r["policy"].file_name
        eligible = r["eligible"]
        reason = r.get("ineligibility_reason", "")
        pid = r["policy"].policy_id
        if eligible:
            if pid == primary_id:
                label = "ELIGIBLE — Selected as Primary Copay Card"
            elif pid == secondary_id:
                label = "ELIGIBLE — Selected as Secondary Foundation Grant"
            else:
                label = "ELIGIBLE — Not selected (alternative program)"
        else:
            label = f"INELIGIBLE — {reason}" if reason else "INELIGIBLE"
        eligibility[fname] = (eligible, label)

    lines = ["[POLICY DOCUMENTS — CITE USING [Source: filename] FORMAT]"]
    for doc_dict in source_docs:
        fname = doc_dict["file_name"]
        _, status_label = eligibility.get(fname, (True, "ELIGIBILITY UNKNOWN"))
        lines += [
            f"\n--- Source: {fname} ---",
            f"ELIGIBILITY: {status_label}",
            doc_dict["excerpt"][:2000],
            f"--- End of {fname} ---",
        ]

    lines.append("[END POLICY DOCUMENTS]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------

class ChatPipelineOrchestrator:

    def __init__(
        self,
        policies_dir: str | Path | None = None,
        api_key: Optional[str] = None,
    ) -> None:
        api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

        self._llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.0,
            google_api_key=api_key,
        )

        # RAG engine indexes all policy docs (cards + grants)
        copay_dir = Path(__file__).parent.parent / "data" / "copay_cards"
        assist_dir = Path(__file__).parent.parent / "data" / "assistance_policies"
        self._kb = HighPrecisionKnowledgeBase(
            docs_dirs=[copay_dir, assist_dir],
            top_k=3,
        )

    def run(
        self,
        patient: dict,
        user_query: str,
        prior_messages: list[dict] | None = None,
    ) -> OrchestratorResponse:
        # Step 1 — Select best stack + run simulation (fully deterministic)
        winning_primary, winning_secondary, simulation_result, card_rankings, grant_rankings = (
            select_best_stack(
                patient,
                copay_cards=list(COPAY_CARD_REGISTRY.values()),
                foundations=list(ASSISTANCE_POLICY_REGISTRY.values()),
            )
        )

        no_eligible_program = (
            winning_primary.policy_id == "NO-POLICY"
            and winning_secondary.policy_id == "NO-POLICY"
        )

        diagnosis = patient.get("diagnosis", "")
        biomarkers = " ".join(patient.get("biomarkers", []))
        drug_code = next(
            (c["billing_code"] for c in patient.get("claims", []) if c["type"] == "drug"),
            "",
        )

        # Step 2 — Targeted doc retrieval
        if no_eligible_program:
            # No winner: retrieve by patient profile so LLM can explain why
            fallback_query = f"{diagnosis} {biomarkers} {drug_code} {user_query}".strip()
            retrieved_docs = self._kb.search(fallback_query)
        else:
            # Guaranteed slots: selected programs + second-best eligible card and grant
            guaranteed_filenames = [
                f for f in [
                    winning_primary.file_name,
                    winning_secondary.file_name,
                    next((r["policy"].file_name for r in card_rankings
                          if r["eligible"] and r["policy"].policy_id != winning_primary.policy_id), None),
                    next((r["policy"].file_name for r in grant_rankings
                          if r["eligible"] and r["policy"].policy_id != winning_secondary.policy_id), None),
                ] if f
            ]
            retrieved_docs = self._kb.search_by_filenames(guaranteed_filenames)

        source_documents = [
            {
                "file_name": doc.metadata.get("source", "unknown"),
                "excerpt": doc.page_content,
                "exact_code_match": doc.metadata.get("exact_code_match", False),
                "retrieval_rank": doc.metadata.get("retrieval_rank", 0),
            }
            for doc in retrieved_docs
        ]

        # Step 3 — Build LLM prompt
        all_rankings = card_rankings + grant_rankings
        simulation_block = _format_simulation_block(simulation_result, card_rankings, grant_rankings)
        policy_block = _format_policy_block(
            source_documents,
            all_rankings=all_rankings,
            primary_id=winning_primary.policy_id,
            secondary_id=winning_secondary.policy_id,
        )

        human_message_content = (
            f"[USER QUESTION]\n{user_query}\n\n"
            f"{simulation_block}\n\n"
            f"{policy_block}\n\n"
            "Respond in bullet points only. "
            "Use exclusively data from [SIMULATION DATA] and [POLICY DOCUMENTS] above. "
            "No external knowledge. No intro or closing sentence. Max 6 bullets."
        )

        # Full conversation history injected — no turn limit
        messages: list = [SystemMessage(content=_SYSTEM_PROMPT)]
        for m in (prior_messages or []):
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=human_message_content))

        # Step 4 — Call LLM for narrative
        try:
            llm_response = self._llm.invoke(messages)
            narrative = llm_response.content
        except Exception as exc:
            narrative = (
                f"[LLM ERROR] The language model could not be reached: {exc}\n\n"
                "The simulation data below was computed deterministically and is accurate."
            )
            return OrchestratorResponse(
                user_query=user_query,
                simulation_result=simulation_result,
                llm_narrative=narrative,
                applied_policy=winning_primary,
                primary_policy=winning_primary,
                secondary_policy=winning_secondary,
                all_card_rankings=card_rankings,
                all_grant_rankings=grant_rankings,
                source_documents=source_documents,
                ineligible_documents=[],
                error=str(exc),
            )

        return OrchestratorResponse(
            user_query=user_query,
            simulation_result=simulation_result,
            llm_narrative=narrative,
            applied_policy=winning_primary,
            primary_policy=winning_primary,
            secondary_policy=winning_secondary,
            all_card_rankings=card_rankings,
            all_grant_rankings=grant_rankings,
            source_documents=source_documents,
            ineligible_documents=[],
            error=None,
        )
