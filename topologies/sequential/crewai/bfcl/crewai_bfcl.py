"""Sequential BFCL runner (CrewAI): analyzer -> inspector -> caller -> verifier crew.

Each task sees the outputs of the earlier tasks as context; the prediction is
the verifier's fenced canonical JSON (else the caller's). The agents'
backstories are the sequential role prompts.
"""

from __future__ import annotations

from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task

from core import prompts, settings
from core.llm import crewai_llm
from core.paths import RESULTS_DIR
from core.tasks import bfcl as task
from core.tasks.bfcl import AST_CATEGORIES, HF_DATASET, extract_canonical, load_instances  # noqa: F401
from core.telemetry import crewai_telemetry, normalize

TOPOLOGY = "sequential"
DEFAULT_OUT_DIR = RESULTS_DIR / "bfcl_sequential"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

_register_model_with_bfcl = task.register_model  # called again by callers that repoint MODEL_ID
_register_model_with_bfcl(MODEL_ID)

_STAGES = ("analyzer", "inspector", "caller", "verifier")


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_llm() -> LLM:
    return crewai_llm(model=MODEL_ID, base_url=VLLM_BASE_URL)


def score_one(function_schemas: list[dict], model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    return task.score_one(function_schemas, model_output, ground_truth, category, MODEL_ID)


def _agent(role: str, title: str, goal: str, model: LLM) -> Agent:
    return Agent(
        role=title,
        goal=goal,
        backstory=_load_prompt(role),
        tools=[],
        llm=model,
        verbose=False,
        allow_delegation=False,
    )


def build_crew(llm: LLM | None = None) -> Crew:
    """The four-agent sequential crew."""
    if llm is None:
        llm = _build_llm()
    analyzer = _agent(
        "analyzer",
        "BFCL Analyzer",
        "Parse the user request to extract intent, entities, and any implicit constraints. Do NOT match against schemas.",
        llm,
    )
    inspector = _agent(
        "inspector",
        "BFCL Inspector",
        "Read the provided function schema(s) and map the analyzer's intent onto a specific function + argument plan. "
        "Do NOT emit the final call.",
        llm,
    )
    caller = _agent(
        "caller",
        "BFCL Caller",
        "Emit the function call(s) per the inspector's plan in BFCL canonical JSON form: a list of single-key dicts "
        '[{"fn_name": {"arg": value, ...}}, ...]. Match the schema types exactly.',
        llm,
    )
    verifier = _agent(
        "verifier",
        "BFCL Verifier",
        "Validate the caller's canonical JSON against the schema: function name exists, required params present, "
        "types coerce. Emit the FINAL canonical JSON (fenced in ```json).",
        llm,
    )

    analyze_task = Task(
        description=(
            "Parse the user request below to extract the intent "
            "(what the user wants done), entities (names, numbers, "
            "dates), and any implicit constraints. Do NOT reference "
            "the schemas yet.\n\n"
            "USER REQUEST:\n{user_request}"
        ),
        expected_output=(
            "A structured summary: (1) intent (one line), (2) bullet "
            "list of entities with their values, (3) any constraints "
            "implied but not stated."
        ),
        agent=analyzer,
    )
    inspect_task = Task(
        description=(
            "Given the Analyzer's summary and the function schema(s) "
            "below, map the intent onto a specific function and "
            "propose argument values. Name the required parameters "
            "and the types they must have.\n\n"
            "USER REQUEST:\n{user_request}\n\n"
            "SCHEMAS:\n{schemas_text}"
        ),
        expected_output=(
            "A plan: which function to call, and for each argument "
            "(required + any relevant optional) the proposed value "
            "and its expected type."
        ),
        agent=inspector,
        context=[analyze_task],
    )
    call_task = Task(
        description=(
            "Emit the call in BFCL canonical JSON. Follow the "
            "Inspector's plan exactly. Output a SINGLE fenced ```json "
            "block containing a list of dicts with one key per dict "
            "where the key is the function name and the value is the "
            "arg dict. For a single call it's a one-element list. For "
            "parallel calls it's a multi-element list. Example:\n"
            '```json\n[{"calculate_triangle_area": {"base": 10, "height": 5}}]\n```\n\n'
            "USER REQUEST:\n{user_request}\n\n"
            "SCHEMAS:\n{schemas_text}"
        ),
        expected_output="A fenced ```json block containing a list of canonical call dicts.",
        agent=caller,
        context=[analyze_task, inspect_task],
    )
    verify_task = Task(
        description=(
            "Validate the Caller's canonical JSON against the "
            "schema(s). Check: (a) each function name appears in the "
            "schemas, (b) every required parameter is present, (c) "
            "each argument's value has the schema's declared type "
            "(coerce only where the schema permits). If correct, "
            "re-emit the SAME JSON. If an error is found, emit a "
            "CORRECTED canonical JSON. Output a SINGLE fenced ```json "
            "block as the FINAL answer.\n\n"
            "USER REQUEST:\n{user_request}\n\n"
            "SCHEMAS:\n{schemas_text}"
        ),
        expected_output="A SINGLE fenced ```json block containing the final canonical call list.",
        agent=verifier,
        context=[analyze_task, inspect_task, call_task],
    )
    return Crew(
        agents=[analyzer, inspector, caller, verifier],
        tasks=[analyze_task, inspect_task, call_task, verify_task],
        process=Process.sequential,
        verbose=False,
    )


def solve(instance: dict) -> dict:
    """Run the crew on one instance.

    Returns ``{"model_output", "by_stage", "raw", "telemetry"}``: the canonical
    calls (or []), every stage's text, the crew's final text and token/call counts.
    """
    crew = build_crew()
    result = crew.kickoff(inputs=task.stage_inputs(instance))
    try:
        stages = {role: result.tasks_output[i].raw for i, role in enumerate(_STAGES)}
    except (AttributeError, IndexError):
        stages = {"analyzer": "", "inspector": "", "caller": "", "verifier": result.raw}
    final = result.raw
    model_output = extract_canonical(stages.get("verifier", "") or final)
    if model_output is None:
        model_output = extract_canonical(stages.get("caller", ""))
    return {
        "model_output": model_output or [],
        "by_stage": stages,
        "raw": final,
        "telemetry": normalize(crewai_telemetry(crew, n_stages=len(stages))),
    }


def run_one(instance: dict, ground_truth: dict, category: str, out_dir: Path) -> dict:
    """Solve and score one instance and write its stage outputs to ``out_dir/traces/<id>.txt``."""
    summary: dict = {"id": instance["id"], "category": category}
    try:
        out = solve(instance)
    except Exception as e:
        return task.solve_failed(summary, e)
    summary["model_output"] = out.get("model_output") or []
    summary["tool_calls"] = len(summary["model_output"])
    summary.update(out.get("telemetry") or {})
    stages = (out.get("by_stage") or {}).items()
    task.write_trace(out_dir, instance["id"], task.sections_trace((role.upper(), text) for role, text in stages))
    return task.add_verdict(summary, score_one, instance, ground_truth, category)


def run_batch(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Score a slice of one subset; writes predictions.jsonl and results.jsonl afresh in ``out_dir``."""
    return task.evaluate(
        run_one, category, limit, offset, only, out_dir or DEFAULT_OUT_DIR, model_id=MODEL_ID, verbose=verbose
    )


def main(argv: list[str] | None = None) -> int:
    return task.cli_main(
        argv,
        description="Sequential-topology BFCL runner (CrewAI 4-stage).",
        run_one=run_one,
        model_id=MODEL_ID,
        default_out_dir=DEFAULT_OUT_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
