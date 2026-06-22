"""Output contract helpers for msg communication-format experiments.

The msg baselines reuse the scorer-facing contracts from the main topology
runners. Communication-format prompts are layered separately in
``communications.communication_formats`` so final HotpotQA/LCB artifacts remain
compatible with the existing evaluators.
"""
from __future__ import annotations

from topologies.output_contracts import (  # noqa: F401
    CONTRACT_GUARD,
    DATASET_CONTRACTS,
    OUTPUT_CONTRACT_VERSION,
    append_output_contract,
    append_output_contract_from_path,
    output_contract,
)

