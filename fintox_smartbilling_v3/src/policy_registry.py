"""
policy_registry.py — PolicyConfig model and FPL reference constants.
POLICY_REGISTRY is no longer defined here — it is built dynamically
by policy_loader.py from the YAML files in data/assistance_policies/.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class PolicyConfig(BaseModel):
    policy_id: str
    file_name: str
    program_name: str
    program_type: str                       # "manufacturer_card" | "foundation_grant"

    # --- Billing & clinical ---
    eligible_billing_codes: list[str]       # empty = any code eligible
    eligible_diagnoses: list[str]           # empty = any diagnosis; OR-matched, case-insensitive
    required_biomarkers: list[str]          # empty = none required; ALL must be present (AND-match)

    # --- Coverage mechanics ---
    covers_facility_fees: bool
    coverage_pct: float                     # 0.0–1.0 fraction of patient responsibility covered
    monthly_max: Optional[float]            # None = no monthly cap
    annual_max: float
    covers_deductible: bool = True          # False = coinsurance/copay only (e.g. PAF)

    # --- Insurance type ---
    requires_commercial_insurance: bool     # True = rejects Medicare/Medicaid/TRICARE/uninsured
    requires_insurance: bool = True         # False = covers uninsured patients too

    # --- Income ---
    income_limit_fpl_pct: Optional[float]   # None = no FPL-based limit; family-size adjusted
    income_limit_absolute: Optional[float]  # None = no flat-dollar limit (e.g. Pfizer $150k)

    # --- Treatment ---
    requires_active_treatment: bool = True  # False = covers patients not yet on treatment


# ---------------------------------------------------------------------------
# FPL 2024 lookup table — used by risk_gate for family-size-adjusted ceiling
# ---------------------------------------------------------------------------
FPL_2024: dict[int, float] = {
    1: 15_060.0,
    2: 20_440.0,
    3: 25_820.0,
    4: 31_200.0,
    5: 36_580.0,
    6: 41_960.0,
    7: 47_340.0,
    8: 52_720.0,
}
FPL_2024_ADDITIONAL_PER_PERSON = 5_380.0   # each person beyond 8
