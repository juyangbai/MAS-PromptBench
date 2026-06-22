"""Protected final-output contracts shared by real topology runners."""
from __future__ import annotations


OUTPUT_CONTRACT_VERSION = 1

CONTRACT_GUARD = (
    "This contract is not optional. Keep reasoning concise enough that the "
    "response reaches the required final artifact. If uncertain, still provide "
    "your best valid final artifact in the required format."
)

DATASET_CONTRACTS = {
    "bfcl": "End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.",
    "gpqa": "End with exactly one final line: Answer: A, Answer: B, Answer: C, or Answer: D.",
    "hotpotqa": "End with exactly one final line: Answer: <short-form>.",
    "math": "Begin with one line containing only \\boxed{...}. After any concise reasoning, end with one line containing only the same \\boxed{...}. The scorer extracts the LAST boxed expression. Use at most 12 concise reasoning lines; do not produce a long derivation that risks truncation.",
    "apps": "End with one fenced ```python``` code block containing the final submitted solution.",
    "apibank": "End with exactly one API call in square brackets, e.g. [ApiName(arg='value')].",
    "lcb": "End with one fenced ```python``` code block containing the final submitted solution.",
    "swe": "End with one fenced ```diff``` block containing the final unified diff patch.",
    "toolhop": "End with exactly one short final answer wrapped as <answer>...</answer>.",
}

PRIMARY_FINAL_ROLES = {"solver", "caller", "coder", "patcher", "planner"}
DECENTRALIZED_FINAL_ROLES = {"debater"}
CENTRALIZED_FINAL_ROLES = {"manager", "manager_r8", "manager_r10"}
CENTRALIZED_FINAL_ROLES_BY_DATASET = {
    "bfcl": {"manager", "caller_worker", "validator_worker"},
    "gpqa": {"manager", "verifier_worker"},
    "hotpotqa": {"manager", "writer_worker"},
    "math": {"manager", "verifier_worker"},
    "apps": {"manager", "coder_worker", "tester_worker", "code_reviewer"},
    "lcb": {"manager", "coder_worker", "tester_worker", "code_reviewer"},
    "swe": {"manager", "tester_worker", "patcher_worker", "commit_summarizer"},
    "toolhop": {"manager", "manager_r8", "manager_r10", "caller_worker", "validator_worker", "answer_normalizer_worker"},
    "apibank": {"manager", "manager_r8", "manager_r10", "caller_worker", "validator_worker", "call_normalizer_worker"},
}
SEQUENTIAL_FINAL_ROLES = {
    "bfcl": {"verifier"},
    "gpqa": {"verifier"},
    "hotpotqa": {"writer"},
    "math": {"verifier"},
    "apps": {"debugger", "code_reviewer"},
    "lcb": {"debugger", "code_reviewer"},
    "swe": {"tester", "commit_summarizer"},
    "toolhop": {"verifier"},
    "apibank": {"verifier"},
}


def output_contract(dataset: str, topology: str, role: str) -> str:
    dataset_key = (dataset or "").strip().lower()
    topology_key = (topology or "").strip().lower()
    role_key = (role or "").strip().lower()
    if topology_key in {"single", "independent"}:
        is_final_role = role_key in PRIMARY_FINAL_ROLES
    elif topology_key in {"decentralized", "decentralized_openai"}:
        is_final_role = role_key in DECENTRALIZED_FINAL_ROLES
    elif topology_key in {"sequential", "sequential_crewai"}:
        is_final_role = role_key in SEQUENTIAL_FINAL_ROLES.get(dataset_key, set())
    elif topology_key in {"centralized", "centralized_autogen"}:
        is_final_role = role_key in CENTRALIZED_FINAL_ROLES_BY_DATASET.get(
            dataset_key, CENTRALIZED_FINAL_ROLES
        )
    else:
        is_final_role = False
    if not is_final_role:
        return ""
    body = DATASET_CONTRACTS.get(dataset_key, "")
    return f"PROTECTED FINAL OUTPUT CONTRACT:\n{body}\n{CONTRACT_GUARD}" if body else ""


def append_output_contract(prompt: str, dataset: str, topology: str, role: str) -> str:
    contract = output_contract(dataset, topology, role)
    if not contract or "PROTECTED FINAL OUTPUT CONTRACT:" in (prompt or ""):
        return prompt
    return (
        contract
        + "\n\n"
        + (prompt or "").rstrip()
        + "\n\nFinal reminder: satisfy the protected final output contract above exactly; "
        "do not stop before emitting the required final artifact."
    )


def append_output_contract_from_path(prompt: str, path: str, role: str) -> str:
    parts = str(path).split("/")
    dataset = ""
    topology = ""
    if "topologies" in parts:
        idx = parts.index("topologies")
        if idx + 1 < len(parts):
            topology = parts[idx + 1]
        known = set(DATASET_CONTRACTS)
        for part in parts[idx + 1:]:
            if part in known:
                dataset = part
                break
    return append_output_contract(prompt, dataset, topology, role)
