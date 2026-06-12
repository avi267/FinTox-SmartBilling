"""
policy_loader.py — Dynamic policy registry built from YAML files.
Reads every *.yaml file in data/assistance_policies/ at startup,
strips the `description` key (prose for RAG), and builds PolicyConfig
objects from the structured fields. POLICY_REGISTRY is the single
runtime authority — the YAML files are the single authoring authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .policy_registry import PolicyConfig

_COPAY_CARDS_DIR = Path(__file__).parent.parent / "data" / "copay_cards"
_ASSISTANCE_DIR  = Path(__file__).parent.parent / "data" / "assistance_policies"


def load_policy_registry(
    policies_dir: Path,
) -> dict[str, PolicyConfig]:
    """
    Parse all *.yaml files in policies_dir and return a dict keyed by policy_id.
    The `description` key is excluded — it is prose for RAG, not a PolicyConfig field.
    Files that fail to parse are skipped with a warning printed to stderr.
    """
    import sys

    registry: dict[str, PolicyConfig] = {}
    for path in sorted(policies_dir.glob("*.yaml")):
        try:
            data: dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            print(f"[policy_loader] WARNING: could not parse {path.name}: {exc}", file=sys.stderr)
            continue

        # Remove RAG-only prose field before passing to Pydantic
        config_fields = {k: v for k, v in data.items() if k != "description"}

        try:
            policy = PolicyConfig(**config_fields)
        except Exception as exc:
            print(f"[policy_loader] WARNING: invalid PolicyConfig in {path.name}: {exc}", file=sys.stderr)
            continue

        registry[policy.policy_id] = policy

    return registry


# Separate registries — manufacturer cards vs foundation grants
COPAY_CARD_REGISTRY:        dict[str, PolicyConfig] = load_policy_registry(_COPAY_CARDS_DIR)
ASSISTANCE_POLICY_REGISTRY: dict[str, PolicyConfig] = load_policy_registry(_ASSISTANCE_DIR)

# Combined registry for RAG engine (needs to search all policy docs)
POLICY_REGISTRY: dict[str, PolicyConfig] = {**COPAY_CARD_REGISTRY, **ASSISTANCE_POLICY_REGISTRY}
