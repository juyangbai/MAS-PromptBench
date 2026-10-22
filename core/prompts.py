"""Role system prompts as the runners send them.

A role's seed prompt is ``configs/prompts/<topology>/<dataset>/<role>.txt``; the
runner may append a dataset nudge, and the role's protected final-output contract
(:mod:`core.output_contracts`) wraps the result.
"""

from __future__ import annotations

import re

from core.output_contracts import append_output_contract
from core.paths import PROMPTS_DIR


def base_role(role: str) -> str:
    """The role without a team-size suffix (``manager_r8`` -> ``manager``), as contracts name it."""
    return re.sub(r"_r\d+$", "", role)


def seed_prompt(topology: str, dataset: str, role: str) -> str:
    """The seed prompt file of ``role``, stripped."""
    return (PROMPTS_DIR / topology / dataset / f"{role}.txt").read_text().strip()


def role_prompt(topology: str, dataset: str, role: str, suffix: str = "") -> str:
    """Seed prompt of ``role`` plus ``suffix``, under the role's protected final-output contract."""
    text = seed_prompt(topology, dataset, role) + suffix
    return append_output_contract(text, dataset, topology, base_role(role))
