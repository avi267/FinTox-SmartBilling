"""
app.py — FinTox-SmartBilling | High-Precision Dual-Input UI Dashboard
Enterprise-grade Streamlit application combining a structured sidebar form
with a conversational main panel and an interactive audit trail.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# ---------------------------------------------------------------------------
# Path setup — allow importing from src/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinTox SmartBilling",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load seed patient data
# ---------------------------------------------------------------------------
@st.cache_data
def _load_patients() -> list[dict]:
    patients_path = ROOT / "data" / "patients.json"
    with open(patients_path, encoding="utf-8") as f:
        return json.load(f)["patients"]


PATIENTS = _load_patients()
PATIENT_NAMES = ["— Select a patient —"] + [p["name"] for p in PATIENTS]


def _get_patient_by_name(name: str) -> dict | None:
    return next((p for p in PATIENTS if p["name"] == name), None)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "messages": [],
        "active_patient": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()

# ---------------------------------------------------------------------------
# Orchestrator factory — cached at server level so LangChain imports and
# BM25 index building only happen once per server start, not per session.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading policy knowledge base…")
def _get_orchestrator(api_key: str):
    from src.orchestrator import ChatPipelineOrchestrator
    return ChatPipelineOrchestrator(api_key=api_key)


# ---------------------------------------------------------------------------
# Sidebar — Structured Input Form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/medical-doctor.png",
        width=64,
    )
    st.title("⚕️ FinTox SmartBilling")
    st.caption("Enterprise Financial Simulation")
    st.divider()

    st.subheader("👤 Patient Profile")
    selected_name = st.selectbox(
        "Load Patient Record",
        options=PATIENT_NAMES,
        help="Auto-populates the form fields below with seed data.",
    )

    seed: dict | None = None
    if selected_name != "— Select a patient —":
        seed = _get_patient_by_name(selected_name)

    def _seed_val(path: list, default):
        obj = seed
        if obj is None:
            return default
        try:
            for key in path:
                obj = obj[key]
            return obj
        except (KeyError, IndexError, TypeError):
            return default

    drug_seed = next((c for c in (seed or {}).get("claims", []) if c["type"] == "drug"), {})
    facility_seed = next((c for c in (seed or {}).get("claims", []) if c["type"] == "facility_fee"), {})

    st.divider()
    st.subheader("📋 Patient Financial Parameters")

    patient_name_field = st.text_input(
        "Patient Name",
        value=_seed_val(["name"], ""),
        placeholder="e.g. Maria",
    )

    diagnosis_field = st.text_input(
        "Diagnosis",
        value=_seed_val(["diagnosis"], ""),
        placeholder="e.g. Stage III Non-Small Cell Lung Cancer (NSCLC)",
        help="Used for diagnosis-based eligibility filtering. Must match the patient's documented oncology diagnosis.",
    )

    insurance_type_field = st.selectbox(
        "Insurance Type",
        options=["commercial", "medicare", "medicaid", "tricare", "chip", "va", "uninsured"],
        index=["commercial", "medicare", "medicaid", "tricare", "chip", "va", "uninsured"].index(
            _seed_val(["insurance_type"], "commercial")
        ),
        help="Manufacturer copay cards require commercial insurance. Government plans disqualify from manufacturer programs.",
    )

    annual_income_field = st.number_input(
        "Annual Household Income ($)",
        min_value=0.0,
        max_value=2_000_000.0,
        value=float(_seed_val(["annual_household_income"], 42000.0)),
        step=1000.0,
        format="%.0f",
        help="Used for FPL-based income eligibility checks. Pfizer also enforces a flat $150,000 cap.",
    )

    family_size_field = st.number_input(
        "Family Size (for FPL calculation)",
        min_value=1,
        max_value=12,
        value=int(_seed_val(["family_size"], 1)),
        step=1,
        help="FPL ceilings are adjusted by family size. E.g. 400% FPL: $60,240 (size 1) vs $124,800 (size 4).",
    )

    treatment_status_field = st.selectbox(
        "Treatment Status",
        options=["active", "initiating", "completed", "surveillance"],
        index=["active", "initiating", "completed", "surveillance"].index(
            _seed_val(["treatment_status"], "active")
        ),
        help="Most programs require active or initiating treatment.",
    )

    biomarkers_field = st.text_input(
        "Biomarkers (comma-separated)",
        value=", ".join(_seed_val(["biomarkers"], [])),
        placeholder="e.g. HER2+, BRCA1, EGFR-",
        help="Required by some programs (e.g. AstraZeneca requires BRCA mutation).",
    )

    remaining_deductible_field = st.number_input(
        "Remaining Deductible ($)",
        min_value=0.0,
        max_value=100_000.0,
        value=float(_seed_val(["insurance", "remaining_deductible"], 500.0)),
        step=50.0,
        format="%.2f",
        help="Patient's outstanding deductible for the current benefit year.",
    )

    coinsurance_pct_field = st.number_input(
        "Coinsurance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(_seed_val(["insurance", "coinsurance_rate"], 0.20)) * 100,
        step=5.0,
        format="%.1f",
        help="Patient's plan coinsurance percentage (e.g. 20 for 20%).",
    )

    st.divider()
    st.subheader("💊 Drug Claim (Assistance Eligible)")

    drug_code_field = st.text_input(
        "Drug Billing Code (HCPCS/CPT)",
        value=drug_seed.get("billing_code", "J9331"),
        placeholder="e.g. J9331",
    ).upper().strip()

    drug_cost_field = st.number_input(
        "Drug Retail Cost ($)",
        min_value=0.0,
        max_value=500_000.0,
        value=float(drug_seed.get("retail_cost", 2500.0)),
        step=100.0,
        format="%.2f",
    )

    st.divider()
    st.subheader("🏥 Facility / Admin Fee Claim")

    facility_cost_field = st.number_input(
        "Facility/Admin Fee Retail Cost ($)",
        min_value=0.0,
        max_value=500_000.0,
        value=float(facility_seed.get("retail_cost", 1500.0)),
        step=100.0,
        format="%.2f",
        help="Clinic administrative or facility fee (not covered by most copay cards).",
    )

    st.divider()

    run_preview = st.button(
        "▶ Run Simulation Preview",
        use_container_width=True,
        type="primary",
        help="Automatically select the best policy stack and compute Path A vs Path B.",
    )

    st.caption(
        "Policy selection is automatic — the engine picks the manufacturer card + foundation grant "
        "stack that minimises patient OOP and preserves the most foundation grant. "
        "All arithmetic is computed by risk_gate.py; the LLM only narrates."
    )


# ---------------------------------------------------------------------------
# Build patient dict from form values
# ---------------------------------------------------------------------------
def _build_patient_from_form() -> dict:
    return {
        "name": patient_name_field or "Patient",
        "diagnosis": diagnosis_field or _seed_val(["diagnosis"], "Oncology Patient"),
        "annual_household_income": annual_income_field,
        "family_size": int(family_size_field),
        "insurance_type": insurance_type_field,
        "treatment_status": treatment_status_field,
        "biomarkers": [b.strip() for b in biomarkers_field.split(",") if b.strip()],
        "insurance": {
            "remaining_deductible": remaining_deductible_field,
            "coinsurance_rate": coinsurance_pct_field / 100.0,
        },
        "claims": [
            {
                "type": "drug",
                "description": "Drug Infusion",
                "billing_code": drug_code_field,
                "retail_cost": drug_cost_field,
                "assistance_program": None,
            },
            {
                "type": "facility_fee",
                "description": "Clinic Administrative & Labor Fee",
                "billing_code": "ADMIN-FEE",
                "retail_cost": facility_cost_field,
                "assistance_program": None,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Policy comparison table — separate tables for cards and grants
# ---------------------------------------------------------------------------
def _render_policy_comparison(
    card_rankings: list[dict],
    grant_rankings: list[dict],
    primary_id: str,
    secondary_id: str,
) -> None:
    import pandas as pd

    with st.expander("🏆 All Assistance Programs — Policy Comparison Table", expanded=True):
        # --- Copay Cards ---
        st.markdown("#### 💊 Manufacturer Copay Cards")
        eligible_cards = [r for r in card_rankings if r["eligible"]]
        ineligible_cards = [r for r in card_rankings if not r["eligible"]]

        if eligible_cards:
            st.markdown("**Eligible cards** — sorted by eligibility")
            card_rows = []
            for r in eligible_cards:
                p = r["policy"]
                card_rows.append({
                    "Program": p.program_name,
                    "Annual Max": f"${p.annual_max:,.0f}",
                    "Covers Deductible": "✅" if p.covers_deductible else "—",
                    "Covers Facility": "✅" if p.covers_facility_fees else "—",
                    "Selected as Primary": "✅ PRIMARY" if p.policy_id == primary_id else "",
                })
            st.dataframe(
                pd.DataFrame(card_rows),
                use_container_width=True,
                hide_index=True,
                height=min(200, 35 + len(card_rows) * 35),
                column_config={"Selected as Primary": st.column_config.TextColumn(width="small")},
            )
        else:
            st.info("No manufacturer copay cards are eligible for this patient (billing code, diagnosis, or insurance type mismatch).")

        if ineligible_cards:
            st.markdown("**Ineligible cards**")
            st.dataframe(
                pd.DataFrame([{
                    "Program": r["policy"].program_name,
                    "Ineligibility Reason": r["ineligibility_reason"],
                } for r in ineligible_cards]),
                use_container_width=True,
                hide_index=True,
                height=min(250, 35 + len(ineligible_cards) * 35),
            )

        st.divider()

        # --- Foundation Grants ---
        st.markdown("#### 🏛 Foundation Grants")
        eligible_grants = [r for r in grant_rankings if r["eligible"]]
        ineligible_grants = [r for r in grant_rankings if not r["eligible"]]

        if eligible_grants:
            st.markdown("**Eligible grants** — sorted by eligibility")
            grant_rows = []
            for r in eligible_grants:
                p = r["policy"]
                grant_rows.append({
                    "Program": p.program_name,
                    "Annual Max": f"${p.annual_max:,.0f}",
                    "Covers Deductible": "✅" if p.covers_deductible else "—",
                    "Covers Facility": "✅" if p.covers_facility_fees else "—",
                    "Income Limit": f"{p.income_limit_fpl_pct:.0f}% FPL" if p.income_limit_fpl_pct else "None",
                    "Selected as Secondary": "✅ SECONDARY" if p.policy_id == secondary_id else "",
                })
            st.dataframe(
                pd.DataFrame(grant_rows),
                use_container_width=True,
                hide_index=True,
                height=min(200, 35 + len(grant_rows) * 35),
                column_config={"Selected as Secondary": st.column_config.TextColumn(width="small")},
            )
        else:
            st.info("No foundation grants are eligible for this patient.")

        if ineligible_grants:
            st.markdown("**Ineligible grants**")
            st.dataframe(
                pd.DataFrame([{
                    "Program": r["policy"].program_name,
                    "Ineligibility Reason": r["ineligibility_reason"],
                } for r in ineligible_grants]),
                use_container_width=True,
                hide_index=True,
                height=min(250, 35 + len(ineligible_grants) * 35),
            )


# ---------------------------------------------------------------------------
# Simulation summary display helper
# ---------------------------------------------------------------------------
def _render_simulation_summary(
    result,
    card_rankings: list[dict] | None = None,
    grant_rankings: list[dict] | None = None,
) -> None:
    primary = result.primary_policy
    secondary = result.secondary_policy

    st.subheader("📊 Comparative Billing Simulation")

    # Selected program stack badge
    if primary.policy_id == "NO-POLICY" and secondary.policy_id == "NO-POLICY":
        st.error(
            "⚠️ No assistance program is currently eligible for this patient. "
            "All programs were disqualified — review the ineligible programs table below for specific reasons. "
            "Do not refer this patient to any program until eligibility criteria are met."
        )
    else:
        primary_label = primary.program_name if primary.policy_id != "NO-POLICY" else "None"
        secondary_label = secondary.program_name if secondary.policy_id != "NO-POLICY" else "None"
        st.markdown(
            f"**Primary (Copay Card):** :green[{primary_label}]  "
            f"| **Secondary (Foundation Grant):** :blue[{secondary_label}]"
        )

    # 4 metric columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Path A — Traditional OOP",
            value=f"${result.path_a_traditional.total_patient_oop:,.2f}",
        )
    with col2:
        st.metric(
            label="Path B — Smart Billing OOP",
            value=f"${result.path_b_smart.total_patient_oop:,.2f}",
            delta=(
                f"-${result.net_savings_smart_vs_traditional:,.2f} saved"
                if result.net_savings_smart_vs_traditional > 0
                else "Same as Path A"
            ),
            delta_color="normal",
        )
    with col3:
        st.metric(
            label="Patient OOP Savings",
            value=f"${result.net_savings_smart_vs_traditional:,.2f}",
        )
    with col4:
        gp = result.foundation_grant_preserved
        if gp > 0:
            col4.metric(
                label="Foundation Grant Preserved",
                value=f"${gp:,.2f}",
                delta=f"Path B saves ${gp:,.2f} of grant",
                delta_color="normal",
                help=(
                    f"Path A would consume ${result.path_a_traditional.total_foundation_absorbs:,.2f} "
                    f"of foundation grant vs Path B's ${result.path_b_smart.total_foundation_absorbs:,.2f}. "
                    f"The ${gp:,.2f} difference is preserved for future treatment visits."
                ),
            )
        elif gp < 0:
            col4.metric(
                label="Grant Invested for $0 OOP",
                value=f"${abs(gp):,.2f}",
                delta=f"Smart billing spends ${abs(gp):,.2f} more grant to eliminate patient OOP",
                delta_color="off",
                help=(
                    f"Path B spends ${abs(gp):,.2f} more foundation grant than Path A, "
                    f"but eliminates ${result.net_savings_smart_vs_traditional:,.2f} in patient out-of-pocket costs. "
                    "The grant investment is worthwhile — the patient saves more than the grant consumes."
                ),
            )
        else:
            col4.metric(
                label="Foundation Grant Impact",
                value="$0.00",
                delta="Same grant usage in both paths",
                delta_color="off",
            )

    # Policy comparison table
    if card_rankings is not None and grant_rankings is not None:
        _render_policy_comparison(
            card_rankings, grant_rankings,
            primary.policy_id, secondary.policy_id,
        )

    # Detailed claim-line breakdown
    with st.expander("📑 Detailed Claim-Line Breakdown", expanded=False):
        tab_a, tab_b = st.tabs(["Path A — Traditional", "Path B — Smart Billing"])

        with tab_a:
            st.markdown(f"**{result.path_a_traditional.path_label}**")
            st.caption(result.path_a_traditional.path_description)
            for i, claim in enumerate(result.path_a_traditional.claim_sequence, 1):
                st.markdown(f"**Claim {i}: {claim.claim_type}** (`{claim.billing_code}`)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retail Cost", f"${claim.retail_cost:,.2f}")
                c2.metric("Deductible Applied", f"${claim.deductible_applied:,.2f}")
                c3.metric("Coinsurance", f"${claim.coinsurance_applied:,.2f}")
                c4.metric("Patient OOP", f"${claim.patient_oop:,.2f}")
                ca, cb, cc = st.columns(3)
                ca.caption(f"💊 Copay Card Covers: **${claim.copay_card_covers:,.2f}**")
                cb.caption(f"🏛 Foundation Covers: **${claim.foundation_covers:,.2f}**")
                insurance_a = round(claim.retail_cost - claim.deductible_applied - claim.coinsurance_applied, 2)
                cc.caption(f"🏥 Insurance Covers: **${insurance_a:,.2f}**")
                st.divider()
            total_insurance_a = sum(
                round(c.retail_cost - c.deductible_applied - c.coinsurance_applied, 2)
                for c in result.path_a_traditional.claim_sequence
            )
            st.success(
                f"Total Patient OOP: **${result.path_a_traditional.total_patient_oop:,.2f}**  |  "
                f"Card Absorbs: **${result.path_a_traditional.total_copay_card_absorbs:,.2f}**  |  "
                f"Foundation Absorbs: **${result.path_a_traditional.total_foundation_absorbs:,.2f}**  |  "
                f"Insurance Covers: **${total_insurance_a:,.2f}**"
            )

        with tab_b:
            st.markdown(f"**{result.path_b_smart.path_label}**")
            st.caption(result.path_b_smart.path_description)
            for i, claim in enumerate(result.path_b_smart.claim_sequence, 1):
                st.markdown(f"**Claim {i}: {claim.claim_type}** (`{claim.billing_code}`)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retail Cost", f"${claim.retail_cost:,.2f}")
                c2.metric("Deductible Applied", f"${claim.deductible_applied:,.2f}")
                c3.metric("Coinsurance", f"${claim.coinsurance_applied:,.2f}")
                c4.metric("Patient OOP", f"${claim.patient_oop:,.2f}")
                ca, cb, cc = st.columns(3)
                ca.caption(f"💊 Copay Card Covers: **${claim.copay_card_covers:,.2f}**")
                cb.caption(f"🏛 Foundation Covers: **${claim.foundation_covers:,.2f}**")
                insurance_b = round(claim.retail_cost - claim.deductible_applied - claim.coinsurance_applied, 2)
                cc.caption(f"🏥 Insurance Covers: **${insurance_b:,.2f}**")
                st.divider()
            total_insurance_b = sum(
                round(c.retail_cost - c.deductible_applied - c.coinsurance_applied, 2)
                for c in result.path_b_smart.claim_sequence
            )
            st.success(
                f"Total Patient OOP: **${result.path_b_smart.total_patient_oop:,.2f}**  |  "
                f"Card Absorbs: **${result.path_b_smart.total_copay_card_absorbs:,.2f}**  |  "
                f"Foundation Absorbs: **${result.path_b_smart.total_foundation_absorbs:,.2f}**  |  "
                f"Insurance Covers: **${total_insurance_b:,.2f}**"
            )


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------
st.title("⚕️ FinTox SmartBilling")
st.caption(
    "Enterprise Health Economics Platform — Deterministic Claim Sequence Optimization for Cancer Patients"
)

st.info(
    "**How to use:** Configure patient parameters in the left sidebar, then ask a question "
    "in the chat below. The engine automatically selects the best manufacturer card + foundation grant "
    "stack and runs the simulation — the AI assistant narrates the findings.",
    icon="ℹ️",
)

# --- Simulation preview ---
if run_preview:
    from src.risk_gate import select_best_stack
    from src.policy_loader import COPAY_CARD_REGISTRY, ASSISTANCE_POLICY_REGISTRY
    patient_data = _build_patient_from_form()
    try:
        winning_primary, winning_secondary, result, card_rankings, grant_rankings = select_best_stack(
            patient_data,
            list(COPAY_CARD_REGISTRY.values()),
            list(ASSISTANCE_POLICY_REGISTRY.values()),
        )
        _render_simulation_summary(result, card_rankings, grant_rankings)
        if result.validation_notes:
            for note in result.validation_notes:
                st.warning(note)
        # Clear chat so the next query starts a fresh session with this patient
        st.session_state["messages"] = []
    except Exception as exc:
        st.error(f"Simulation error: {exc}")

st.divider()

def _render_audit_trail(
    source_documents: list,
    selected_filenames: list[str],
    key_prefix: str,
) -> None:
    """Render the knowledge base audit trail — filtered to selected program docs only."""
    docs_to_show = [
        s for s in source_documents
        if s.get("file_name", "") in selected_filenames
    ]

    if docs_to_show:
        st.divider()
        st.markdown("##### 📚 Knowledge Base Audit Trail")
        for src in docs_to_show:
            file_name = src.get("file_name", "unknown")
            exact_badge = " ✅ Exact Code Match" if src.get("exact_code_match") else ""
            with st.expander(
                f"🔍 Audit Trail Verified: {file_name}{exact_badge}",
                expanded=False,
            ):
                meta_lines = f"**Source file:** `{file_name}`  \n**Retrieval rank:** {src.get('retrieval_rank', '—')}"
                if src.get("exact_code_match"):
                    meta_lines += "  \n**Exact billing code match:** ✅"
                st.markdown(meta_lines)
                st.divider()
                st.text_area(
                    "Raw policy text used by the model:",
                    value=src.get("excerpt", ""),
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"{key_prefix}_{file_name}_{src.get('retrieval_rank')}",
                )


# --- Chat history ---
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            _render_audit_trail(
                source_documents=msg.get("source_documents", []),
                selected_filenames=msg.get("selected_filenames", []),
                key_prefix=f"hist_{msg_idx}",
            )

# --- Chat input ---
user_input = st.chat_input(
    "Ask about this patient's billing options, copay card rules, or savings opportunities…"
)

if user_input:
    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("GOOGLE_API_KEY not found. Please set it in the .env file.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    patient_data = _build_patient_from_form()

    # Ext 5 — Collect prior user turns for multi-turn retrieval context
    prior_user_queries = [
        m["content"]
        for m in st.session_state.messages
        if m["role"] == "user"
    ]

    # First user message in this session gets the full audit trail
    is_first_query = len(prior_user_queries) == 1

    with st.chat_message("assistant"):
        with st.spinner("Selecting best policy stack + running simulation + AI narrative…"):
            try:
                orchestrator = _get_orchestrator(os.environ.get("GOOGLE_API_KEY", ""))
                response = orchestrator.run(
                    patient=patient_data,
                    user_query=user_input,
                    prior_user_queries=prior_user_queries,
                )
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

        st.markdown(response.llm_narrative)

        if response.error:
            st.warning(f"Note: LLM call encountered an error — {response.error}")

        # Selected program filenames — used for follow-up audit trail filtering
        selected_filenames = [
            f for f in [
                response.primary_policy.file_name,
                response.secondary_policy.file_name,
            ] if f
        ]

        msg_key = len(st.session_state.messages)
        _render_audit_trail(
            source_documents=response.source_documents,
            selected_filenames=selected_filenames,
            key_prefix=f"audit_{msg_key}",
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.llm_narrative,
        "source_documents": response.source_documents,
        "ineligible_documents": response.ineligible_documents,
        "is_first_query": is_first_query,
        "selected_filenames": selected_filenames,
    })
