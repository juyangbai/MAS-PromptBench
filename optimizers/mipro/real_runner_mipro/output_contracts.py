"""Protected final-output contracts for real-runner MIPRO adapters."""
from __future__ import annotations


OUTPUT_CONTRACT_VERSION = 2

CONTRACT_GUARD = (
    "This contract is not optional. Keep reasoning concise enough that the "
    "response reaches the required final artifact. If uncertain, still provide "
    "your best valid final artifact in the required format."
)


DATASET_CONTRACTS = {
    "bfcl": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "End the final response with exactly one fenced ```json``` block. "
        "The JSON must be a non-empty list of canonical function-call dicts, "
        "for example [{\"function_name\": {\"arg\": \"value\"}}]."
    ),
    "gpqa": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "End the final response with exactly one final line: Answer: A, "
        "Answer: B, Answer: C, or Answer: D."
    ),
    "hotpotqa": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "End the final response with exactly one final line: "
        "Answer: <short-form>. The short-form should be minimal, such as a "
        "name, year, place, noun phrase, yes, or no."
    ),
    "math": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "Begin the final response with one line containing only \\boxed{...}. "
        "After any concise reasoning, end the final response with one line "
        "containing only the same \\boxed{...}. The scorer extracts the LAST "
        "\\boxed{...} expression. Use at most 12 concise reasoning lines; do "
        "not produce a long derivation that risks truncation."
    ),
    "apps": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "End the final response with exactly one fenced ```python``` code "
        "block containing the final submitted solution. Keep reasoning very "
        "brief and emit the complete code block before any optional notes."
    ),
    "apibank": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "End the final response with exactly one bracketed API call and no "
        "surrounding prose on that final line, for example "
        "[ApiName(arg='value', other_arg=123)]. The call must use the exact API "
        "name and keyword argument names expected by API-Bank."
    ),
    "toolhop": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "End the final response with exactly one short final answer wrapped as "
        "<answer>...</answer>. Do not put the final answer only in tool output, "
        "JSON, or explanatory prose."
    ),
    "lcb": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "End the final response with exactly one fenced ```python``` code "
        "block containing the final submitted solution. Keep reasoning very "
        "brief and emit the complete code block before any optional notes."
    ),
    "swe": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "End the final response with exactly one fenced ```diff``` block "
        "containing the final unified diff patch. Keep reasoning very brief "
        "and emit the complete patch before any optional notes."
    ),
    "travel": (
        "PROTECTED FINAL OUTPUT CONTRACT:\n"
        "End the final response with exactly one fenced ```json``` block "
        "containing a non-empty JSON list of per-day plan dictionaries. "
        "Each dictionary must include at least these keys: days, "
        "current_city, transportation. Prefer also including breakfast, "
        "lunch, dinner, attraction, and accommodation."
    ),
}


PRIMARY_FINAL_ROLES = {"solver", "caller", "coder", "patcher", "planner"}
DECENTRALIZED_FINAL_ROLES = {"debater"}
CENTRALIZED_FINAL_ROLES = {"manager"}
CENTRALIZED_FINAL_ROLES_BY_DATASET = {
    "bfcl": {"manager", "caller_worker", "validator_worker"},
    "gpqa": {"manager", "verifier_worker"},
    "hotpotqa": {"manager", "writer_worker"},
    "math": {"manager", "verifier_worker"},
    "apps": {"manager", "coder_worker", "tester_worker", "code_reviewer"},
    "lcb": {"manager", "coder_worker", "tester_worker", "code_reviewer"},
    "swe": {"manager", "tester_worker", "patcher_worker", "commit_summarizer"},
    "apibank": {"manager", "caller_worker", "validator_worker", "call_normalizer_worker"},
    "toolhop": {"manager", "caller_worker", "validator_worker", "answer_normalizer_worker"},
    "travel": {"manager", "itinerary_worker", "validator_worker", "finalizer"},
}
SEQUENTIAL_FINAL_ROLES = {
    "bfcl": {"verifier"},
    "gpqa": {"verifier"},
    "hotpotqa": {"writer"},
    "math": {"verifier"},
    "apps": {"debugger", "code_reviewer"},
    "lcb": {"debugger", "code_reviewer"},
    "swe": {"patcher", "tester", "commit_summarizer"},
    "apibank": {"verifier", "call_normalizer"},
    "toolhop": {"verifier", "answer_normalizer"},
    "travel": {"finalizer"},
}


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def output_contract(dataset: str, topology: str, role: str) -> str:
    """Return the protected final-format contract for a dataset/topology role."""
    dataset_key = _norm(dataset)
    topology_key = _norm(topology)
    role_key = _norm(role)
    if role_key.startswith("manager_r") and role_key.removeprefix("manager_r").isdigit():
        role_key = "manager"
    if dataset_key not in DATASET_CONTRACTS:
        return ""
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
    return DATASET_CONTRACTS[dataset_key] + "\n" + CONTRACT_GUARD


def append_output_contract(prompt: str, dataset: str, topology: str, role: str) -> str:
    """Append a dataset contract idempotently at execution time."""
    contract = output_contract(dataset, topology, role)
    if not contract:
        return prompt
    text = prompt or ""
    if "PROTECTED FINAL OUTPUT CONTRACT:" in text:
        return text
    return (
        contract
        + "\n\n"
        + text.rstrip()
        + "\n\nFinal reminder: satisfy the protected final output contract above exactly; "
        "do not stop before emitting the required final artifact."
    )
