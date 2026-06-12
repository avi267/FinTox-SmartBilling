"""
risk_gate.py — Advanced Structured Financial Validation
Deterministic, isolated Python computing layer for claim sequence optimization.
No LLM involvement in any arithmetic. All values are Pydantic-validated.

Stacking model (Flatiron-aligned):
  Primary:   Manufacturer copay card — covers drug deductible + coinsurance first.
             Legally cannot cover facility fees.
  Secondary: Foundation grant — covers any remaining patient balance
             (facility coinsurance, uncovered drug costs).
  Metric:    Foundation Grant Preserved = Path A grant spend - Path B grant spend.
             When patient OOP = $0 in both paths, smart billing is still better because
             it consumes less of the foundation's finite annual grant.
"""

from __future__ import annotations
import re
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

from .policy_registry import PolicyConfig, FPL_2024, FPL_2024_ADDITIONAL_PER_PERSON
from .policy_loader import COPAY_CARD_REGISTRY, ASSISTANCE_POLICY_REGISTRY


# ---------------------------------------------------------------------------
# Strongly-typed result models
# ---------------------------------------------------------------------------

class ClaimLineResult(BaseModel):
    claim_type: str
    billing_code: str
    retail_cost: float
    deductible_applied: float
    coinsurance_applied: float
    copay_card_covers: float
    foundation_covers: float = 0.0
    patient_oop: float

    @field_validator(
        "retail_cost", "deductible_applied", "coinsurance_applied",
        "copay_card_covers", "foundation_covers", "patient_oop"
    )
    @classmethod
    def no_negative_values(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"Financial integrity violation: negative value detected ({v}). "
                "All billing outputs must be >= $0.00."
            )
        return round(v, 2)


class BillingPathResult(BaseModel):
    path_label: str
    path_description: str
    claim_sequence: list[ClaimLineResult]
    total_patient_oop: float
    total_copay_card_absorbs: float
    total_foundation_absorbs: float = 0.0

    @field_validator("total_patient_oop", "total_copay_card_absorbs", "total_foundation_absorbs")
    @classmethod
    def no_negative_totals(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"Financial integrity violation: negative total detected ({v})."
            )
        return round(v, 2)


class ComparativeBillingResult(BaseModel):
    patient_name: str
    diagnosis: str
    remaining_deductible_at_start: float
    coinsurance_rate: float
    applied_policy: PolicyConfig          # = primary_policy for backwards compat
    primary_policy: PolicyConfig          # manufacturer card (_NO_POLICY if none)
    secondary_policy: PolicyConfig        # foundation grant (_NO_POLICY if none)
    path_a_traditional: BillingPathResult
    path_b_smart: BillingPathResult
    net_savings_smart_vs_traditional: float
    foundation_grant_preserved: float = 0.0
    simulation_status: str = "SUCCESS"
    validation_notes: list[str] = []

    @model_validator(mode="after")
    def validate_and_compute(self) -> "ComparativeBillingResult":
        expected = round(
            self.path_a_traditional.total_patient_oop
            - self.path_b_smart.total_patient_oop,
            2,
        )
        if abs(expected - self.net_savings_smart_vs_traditional) > 0.01:
            raise ValueError(
                f"Savings integrity check failed: "
                f"expected {expected}, got {self.net_savings_smart_vs_traditional}"
            )
        self.foundation_grant_preserved = round(
            self.path_a_traditional.total_foundation_absorbs
            - self.path_b_smart.total_foundation_absorbs,
            2,
        )
        return self


# ---------------------------------------------------------------------------
# Eligibility check
# ---------------------------------------------------------------------------

_GOVT_INSURANCE_TYPES = {"medicare", "medicaid", "tricare", "chip", "va", "indian_health"}


def _fpl_ceiling(policy: PolicyConfig, family_size: int) -> float:
    """Return the dollar income ceiling for this policy, adjusted for family size."""
    if policy.income_limit_fpl_pct is None:
        return float("inf")
    size = max(1, min(family_size, 8))
    base = FPL_2024.get(size, FPL_2024[8] + (size - 8) * FPL_2024_ADDITIONAL_PER_PERSON)
    return base * (policy.income_limit_fpl_pct / 100.0)


def _is_policy_eligible(patient: dict, policy: PolicyConfig) -> tuple[bool, str]:
    """
    Return (eligible: bool, reason: str).
    reason is empty string when eligible; a human-readable explanation when not.
    """

    # 1. Must have a drug claim
    drug_claim = next((c for c in patient["claims"] if c["type"] == "drug"), None)
    if drug_claim is None:
        return False, "No drug claim found on patient record."

    drug_code = drug_claim["billing_code"].upper()
    diagnosis_lower = patient.get("diagnosis", "").lower()
    insurance_type = patient.get("insurance_type", "commercial").lower()
    biomarkers = [b.upper() for b in patient.get("biomarkers", [])]
    treatment_status = patient.get("treatment_status", "active").lower()
    annual_income = float(patient.get("annual_household_income", 0.0))
    family_size = int(patient.get("family_size", 1))

    # 2. Billing code match (applies to manufacturer cards that specify codes)
    if policy.eligible_billing_codes:
        allowed = [c.upper() for c in policy.eligible_billing_codes]
        if drug_code not in allowed:
            return False, (
                f"Billing code {drug_code} not covered. "
                f"This program covers: {', '.join(allowed)}."
            )

    # 3. Diagnosis keyword match (OR — at least one keyword must appear as a whole word)
    if policy.eligible_diagnoses:
        if not any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', diagnosis_lower) for kw in policy.eligible_diagnoses):
            return False, (
                f"Diagnosis '{patient.get('diagnosis', '')}' does not match any "
                f"eligible condition for this program "
                f"({', '.join(policy.eligible_diagnoses)})."
            )

    # 4. Required biomarkers (AND — every listed marker must appear in patient list)
    if policy.required_biomarkers:
        missing = [
            m for m in policy.required_biomarkers
            if not any(m.upper() in b for b in biomarkers)
        ]
        if missing:
            return False, (
                f"Required biomarker(s) not documented: {', '.join(missing)}. "
                f"Patient biomarkers on file: {', '.join(biomarkers) or 'none'}."
            )

    # 5. Commercial insurance required (manufacturer cards)
    if policy.requires_commercial_insurance:
        if insurance_type in _GOVT_INSURANCE_TYPES or insurance_type == "uninsured":
            return False, (
                f"This manufacturer program requires commercial insurance. "
                f"Patient insurance type: {insurance_type}."
            )

    # 6. Active insurance required (most foundation grants need someone to bill)
    if policy.requires_insurance:
        if insurance_type == "uninsured":
            return False, "Patient must have active insurance coverage to enroll."

    # 7. Active treatment status
    if policy.requires_active_treatment:
        if treatment_status not in {"active", "initiating"}:
            return False, (
                f"Patient treatment status '{treatment_status}' does not meet "
                f"the active/initiating treatment requirement."
            )

    # 8. FPL-adjusted income ceiling
    if policy.income_limit_fpl_pct is not None:
        ceiling = _fpl_ceiling(policy, family_size)
        if annual_income > ceiling:
            return False, (
                f"Household income ${annual_income:,.0f} exceeds the "
                f"{policy.income_limit_fpl_pct:.0f}% FPL ceiling "
                f"of ${ceiling:,.0f} for a family of {family_size}."
            )

    # 9. Absolute income ceiling (e.g. Pfizer's $150,000 flat cap)
    if policy.income_limit_absolute is not None:
        if annual_income > policy.income_limit_absolute:
            return False, (
                f"Household income ${annual_income:,.0f} exceeds the program's "
                f"absolute income cap of ${policy.income_limit_absolute:,.0f}."
            )

    return True, ""


# ---------------------------------------------------------------------------
# Zero-benefit baseline
# ---------------------------------------------------------------------------

_NO_POLICY = PolicyConfig(
    policy_id="NO-POLICY",
    file_name="",
    program_name="No Assistance Program",
    program_type="none",
    eligible_billing_codes=[],
    eligible_diagnoses=[],
    required_biomarkers=[],
    covers_facility_fees=False,
    coverage_pct=0.0,
    monthly_max=None,
    annual_max=0.0,
    covers_deductible=False,
    requires_commercial_insurance=False,
    requires_insurance=False,
    income_limit_fpl_pct=None,
    income_limit_absolute=None,
    requires_active_treatment=False,
)


# ---------------------------------------------------------------------------
# Core simulation primitives
# ---------------------------------------------------------------------------

def _apply_card(
    patient_gross: float,
    policy: PolicyConfig,
    card_spent_so_far: float,
    deductible_component: float,
) -> float:
    """
    Calculate how much the assistance program covers on a single claim line,
    respecting annual maximum, monthly maximum, and deductible coverage rules.
    Returns the dollar amount covered (>= 0).
    """
    if not policy.covers_deductible:
        # PAF-style: covers only coinsurance, not deductible
        coverable = patient_gross - deductible_component
    else:
        coverable = patient_gross

    coverable = max(0.0, coverable)

    if policy.monthly_max is not None:
        coverable = min(coverable, policy.monthly_max)

    remaining_annual = max(0.0, policy.annual_max - card_spent_so_far)
    coverable = min(coverable, remaining_annual)

    return round(coverable * policy.coverage_pct, 2)


def _apply_stacked(
    claim_gross: float,
    deductible_component: float,
    copay_card: PolicyConfig,
    foundation: PolicyConfig,
    card_spent: float,
    grant_spent: float,
    is_facility_claim: bool,
) -> tuple[float, float, float]:
    """
    Apply manufacturer card then foundation grant to a single claim.

    Step 1 — Manufacturer card covers the claim (facility only if covers_facility_fees=True).
    Step 2 — Foundation grant covers remaining patient balance (facility only if covers_facility_fees=True).

    Returns (card_covers, foundation_covers, patient_oop).
    """
    # Step 1: Manufacturer card
    if is_facility_claim and not copay_card.covers_facility_fees:
        card_covers = 0.0
    else:
        card_covers = _apply_card(claim_gross, copay_card, card_spent, deductible_component)

    remaining_oop = round(claim_gross - card_covers, 2)

    # Track how much deductible the card actually absorbed
    if copay_card.covers_deductible and card_covers > 0:
        card_ded_absorbed = min(deductible_component, card_covers)
    else:
        card_ded_absorbed = 0.0
    remaining_ded = max(0.0, deductible_component - card_ded_absorbed)

    # Step 2: Foundation grant on remaining patient balance
    if is_facility_claim and not foundation.covers_facility_fees:
        foundation_covers = 0.0
    else:
        foundation_covers = _apply_card(remaining_oop, foundation, grant_spent, remaining_ded)

    patient_oop = round(remaining_oop - foundation_covers, 2)
    return card_covers, foundation_covers, patient_oop


# ---------------------------------------------------------------------------
# Stacked billing path simulation
# ---------------------------------------------------------------------------

def calculate_stacked_billing_paths(
    patient: dict,
    copay_card: PolicyConfig,
    foundation: PolicyConfig,
) -> ComparativeBillingResult:
    """
    Simulate Path A and Path B using a stacked manufacturer card + foundation grant.

    Path A (Traditional): Facility fee billed first, drug second.
    Path B (Smart Billing): Drug billed first (card absorbs deductible), facility second.

    The manufacturer card covers drug OOP first (legally cannot touch facility fees).
    The foundation grant covers any remaining patient balance after the card.
    """
    name: str = patient["name"]
    diagnosis: str = patient["diagnosis"]
    deductible: float = float(patient["insurance"]["remaining_deductible"])
    coinsurance_rate: float = float(patient["insurance"]["coinsurance_rate"])

    drug_claim = next(c for c in patient["claims"] if c["type"] == "drug")
    facility_claim = next(c for c in patient["claims"] if c["type"] == "facility_fee")

    drug_cost: float = float(drug_claim["retail_cost"])
    drug_code: str = drug_claim["billing_code"]
    facility_cost: float = float(facility_claim["retail_cost"])
    facility_code: str = facility_claim["billing_code"]

    validation_notes: list[str] = []

    # -----------------------------------------------------------------------
    # PATH A — Traditional (Facility first, then drug)
    # -----------------------------------------------------------------------
    card_spent_a = 0.0
    grant_spent_a = 0.0

    # A1: Facility claim
    a_fac_ded = round(min(deductible, facility_cost), 2)
    a_fac_rem = round(facility_cost - a_fac_ded, 2)
    a_fac_coins = round(a_fac_rem * coinsurance_rate, 2)
    a_fac_gross = round(a_fac_ded + a_fac_coins, 2)
    a_fac_card, a_fac_grant, a_fac_oop = _apply_stacked(
        a_fac_gross, a_fac_ded, copay_card, foundation,
        card_spent_a, grant_spent_a, is_facility_claim=True,
    )
    card_spent_a += a_fac_card
    grant_spent_a += a_fac_grant
    a_ded_after_fac = round(deductible - a_fac_ded, 2)

    a_fac_line = ClaimLineResult(
        claim_type="Facility / Admin Fee",
        billing_code=facility_code,
        retail_cost=facility_cost,
        deductible_applied=a_fac_ded,
        coinsurance_applied=a_fac_coins,
        copay_card_covers=a_fac_card,
        foundation_covers=a_fac_grant,
        patient_oop=a_fac_oop,
    )

    # A2: Drug claim — deductible partially (or fully) used by facility
    a_drug_ded = round(min(a_ded_after_fac, drug_cost), 2)
    a_drug_rem = round(drug_cost - a_drug_ded, 2)
    a_drug_coins = round(a_drug_rem * coinsurance_rate, 2)
    a_drug_gross = round(a_drug_ded + a_drug_coins, 2)
    a_drug_card, a_drug_grant, a_drug_oop = _apply_stacked(
        a_drug_gross, a_drug_ded, copay_card, foundation,
        card_spent_a, grant_spent_a, is_facility_claim=False,
    )
    grant_spent_a += a_drug_grant

    a_drug_line = ClaimLineResult(
        claim_type="Drug Infusion",
        billing_code=drug_code,
        retail_cost=drug_cost,
        deductible_applied=a_drug_ded,
        coinsurance_applied=a_drug_coins,
        copay_card_covers=a_drug_card,
        foundation_covers=a_drug_grant,
        patient_oop=a_drug_oop,
    )

    path_a_total_oop = round(a_fac_oop + a_drug_oop, 2)
    path_a_card_absorbs = round(a_fac_card + a_drug_card, 2)
    path_a_grant_absorbs = round(a_fac_grant + a_drug_grant, 2)

    # Path A description
    if copay_card.policy_id == "NO-POLICY":
        a_card_text = "No manufacturer copay card available."
    elif copay_card.covers_facility_fees:
        a_card_text = (
            f"{copay_card.program_name} covers facility fees; card absorbs "
            f"{'deductible and ' if copay_card.covers_deductible else ''}coinsurance."
        )
    else:
        a_card_text = (
            f"{copay_card.program_name} excludes facility fees; patient absorbs "
            f"full deductible and coinsurance on facility claim out-of-pocket."
        )
    a_grant_text = (
        f" {foundation.program_name} covers remaining patient balance."
        if foundation.policy_id != "NO-POLICY" else ""
    )

    path_a = BillingPathResult(
        path_label="Path A — Traditional Billing",
        path_description=(
            f"Facility/admin fee submitted first. {a_card_text}{a_grant_text}"
            f" Drug claim arrives second with deductible "
            f"{'partially ' if a_ded_after_fac > 0 else ''}exhausted."
        ),
        claim_sequence=[a_fac_line, a_drug_line],
        total_patient_oop=path_a_total_oop,
        total_copay_card_absorbs=path_a_card_absorbs,
        total_foundation_absorbs=path_a_grant_absorbs,
    )

    # -----------------------------------------------------------------------
    # PATH B — Smart Billing (Drug first, then facility)
    # -----------------------------------------------------------------------
    card_spent_b = 0.0
    grant_spent_b = 0.0

    # B1: Drug claim first — card absorbs deductible + coinsurance
    b_drug_ded = round(min(deductible, drug_cost), 2)
    b_drug_rem = round(drug_cost - b_drug_ded, 2)
    b_drug_coins = round(b_drug_rem * coinsurance_rate, 2)
    b_drug_gross = round(b_drug_ded + b_drug_coins, 2)
    b_drug_card, b_drug_grant, b_drug_oop = _apply_stacked(
        b_drug_gross, b_drug_ded, copay_card, foundation,
        card_spent_b, grant_spent_b, is_facility_claim=False,
    )
    card_spent_b += b_drug_card
    grant_spent_b += b_drug_grant
    b_ded_after_drug = round(deductible - b_drug_ded, 2)

    b_drug_line = ClaimLineResult(
        claim_type="Drug Infusion",
        billing_code=drug_code,
        retail_cost=drug_cost,
        deductible_applied=b_drug_ded,
        coinsurance_applied=b_drug_coins,
        copay_card_covers=b_drug_card,
        foundation_covers=b_drug_grant,
        patient_oop=b_drug_oop,
    )

    # B2: Facility claim — deductible already exhausted by drug
    b_fac_ded = round(min(b_ded_after_drug, facility_cost), 2)
    b_fac_rem = round(facility_cost - b_fac_ded, 2)
    b_fac_coins = round(b_fac_rem * coinsurance_rate, 2)
    b_fac_gross = round(b_fac_ded + b_fac_coins, 2)
    b_fac_card, b_fac_grant, b_fac_oop = _apply_stacked(
        b_fac_gross, b_fac_ded, copay_card, foundation,
        card_spent_b, grant_spent_b, is_facility_claim=True,
    )
    grant_spent_b += b_fac_grant

    b_fac_line = ClaimLineResult(
        claim_type="Facility / Admin Fee",
        billing_code=facility_code,
        retail_cost=facility_cost,
        deductible_applied=b_fac_ded,
        coinsurance_applied=b_fac_coins,
        copay_card_covers=b_fac_card,
        foundation_covers=b_fac_grant,
        patient_oop=b_fac_oop,
    )

    path_b_total_oop = round(b_drug_oop + b_fac_oop, 2)
    path_b_card_absorbs = round(b_drug_card + b_fac_card, 2)
    path_b_grant_absorbs = round(b_drug_grant + b_fac_grant, 2)

    # Path B description
    if copay_card.policy_id == "NO-POLICY":
        b_card_text = "No manufacturer card; patient owes full drug deductible and coinsurance."
    else:
        b_card_text = (
            f"{copay_card.program_name} absorbs ${b_drug_card:,.2f} on the drug claim "
            f"({'deductible + coinsurance' if copay_card.covers_deductible else 'coinsurance only'})."
        )
    b_grant_text = (
        f" {foundation.program_name} covers ${b_fac_grant:,.2f} remaining facility balance."
        if foundation.policy_id != "NO-POLICY" else ""
    )

    path_b = BillingPathResult(
        path_label="Path B — Smart Billing (Optimized)",
        path_description=(
            f"Drug claim ({drug_code}) submitted first. {b_card_text}{b_grant_text}"
            f" Facility arrives with deductible "
            f"{'fully ' if b_ded_after_drug == 0 else 'partially '}satisfied."
        ),
        claim_sequence=[b_drug_line, b_fac_line],
        total_patient_oop=path_b_total_oop,
        total_copay_card_absorbs=path_b_card_absorbs,
        total_foundation_absorbs=path_b_grant_absorbs,
    )

    net_savings = round(path_a_total_oop - path_b_total_oop, 2)

    return ComparativeBillingResult(
        patient_name=name,
        diagnosis=diagnosis,
        remaining_deductible_at_start=deductible,
        coinsurance_rate=coinsurance_rate,
        applied_policy=copay_card,   # primary for backwards compat
        primary_policy=copay_card,
        secondary_policy=foundation,
        path_a_traditional=path_a,
        path_b_smart=path_b,
        net_savings_smart_vs_traditional=net_savings,
        simulation_status="SUCCESS",
        validation_notes=validation_notes,
    )


# ---------------------------------------------------------------------------
# Stack selection
# ---------------------------------------------------------------------------

def select_best_stack(
    patient: dict,
    copay_cards: list[PolicyConfig] | None = None,
    foundations: list[PolicyConfig] | None = None,
) -> tuple[PolicyConfig, PolicyConfig, ComparativeBillingResult, list[dict], list[dict]]:
    """
    Evaluate all eligible (copay_card, foundation) combinations and return the
    stack that minimises Path B patient OOP, then minimises foundation grant spend.

    Returns:
        (best_primary, best_secondary, best_result, card_rankings, grant_rankings)
        card_rankings: per-card eligibility list [{policy, eligible, ineligibility_reason, is_selected}]
        grant_rankings: per-grant eligibility list [{policy, eligible, ineligibility_reason, is_selected}]
    """
    if copay_cards is None:
        copay_cards = list(COPAY_CARD_REGISTRY.values())
    if foundations is None:
        foundations = list(ASSISTANCE_POLICY_REGISTRY.values())

    # Evaluate eligibility of each card and each foundation independently
    card_rankings: list[dict] = []
    grant_rankings: list[dict] = []
    eligible_cards: list[PolicyConfig] = []
    eligible_grants: list[PolicyConfig] = []

    for card in copay_cards:
        eligible, reason = _is_policy_eligible(patient, card)
        card_rankings.append({
            "policy": card,
            "eligible": eligible,
            "ineligibility_reason": reason,
            "is_selected": False,
        })
        if eligible:
            eligible_cards.append(card)

    for grant in foundations:
        eligible, reason = _is_policy_eligible(patient, grant)
        grant_rankings.append({
            "policy": grant,
            "eligible": eligible,
            "ineligibility_reason": reason,
            "is_selected": False,
        })
        if eligible:
            eligible_grants.append(grant)

    # Build candidate stacks — fall back to _NO_POLICY if one side has no eligible programs
    card_candidates = eligible_cards if eligible_cards else [_NO_POLICY]
    grant_candidates = eligible_grants if eligible_grants else [_NO_POLICY]

    stack_results: list[dict] = []
    for card in card_candidates:
        for grant in grant_candidates:
            result = calculate_stacked_billing_paths(patient, card, grant)
            stack_results.append({
                "primary": card,
                "secondary": grant,
                "result": result,
                "path_b_oop": result.path_b_smart.total_patient_oop,
                "path_b_grant_spend": result.path_b_smart.total_foundation_absorbs,
            })

    # Rank: Path B OOP ascending → foundation grant spend ascending → combined annual_max descending
    stack_results.sort(key=lambda x: (
        x["path_b_oop"],
        x["path_b_grant_spend"],
        -(x["primary"].annual_max + x["secondary"].annual_max),
    ))

    best = stack_results[0]
    best_primary = best["primary"]
    best_secondary = best["secondary"]
    best_result = best["result"]

    # Mark selected policies in rankings
    for r in card_rankings:
        if r["policy"].policy_id == best_primary.policy_id:
            r["is_selected"] = True
    for r in grant_rankings:
        if r["policy"].policy_id == best_secondary.policy_id:
            r["is_selected"] = True

    return best_primary, best_secondary, best_result, card_rankings, grant_rankings
