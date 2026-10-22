"""Sequential MATH runner (CrewAI): decomposer -> computer -> checker -> verifier crew.

Each task sees the outputs of the earlier tasks as context; the answer is the
verifier's boxed answer. The agents' backstories are the sequential role prompts.
"""

from __future__ import annotations

from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from core import cli, prompts, settings
from core.batch import attempt
from core.calculator import CALCULATOR_DOC_DECIMAL_PI, make_calculator
from core.llm import crewai_llm
from core.tasks import math as task
from core.tasks.math import exact_match_score, extract_answer, extract_boxed, is_equiv, load_instances  # noqa: F401
from core.telemetry import crewai_telemetry, normalize

TOPOLOGY = "sequential"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

calculator = tool("calculator")(make_calculator(CALCULATOR_DOC_DECIMAL_PI))


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_llm() -> LLM:
    return crewai_llm(model=MODEL_ID, base_url=VLLM_BASE_URL)


def _agent(role: str, title: str, goal: str, tools: list, model: LLM) -> Agent:
    return Agent(
        role=title,
        goal=goal,
        backstory=_load_prompt(role),
        tools=tools,
        llm=model,
        verbose=False,
        allow_delegation=False,
    )


def build_crew(llm: LLM | None = None) -> Crew:
    """The four-agent sequential crew."""
    if llm is None:
        llm = _build_llm()
    decomposer = _agent(
        "decomposer",
        "Math Decomposer",
        "Break the math problem into a short ordered list of computational sub-steps. Do NOT compute anything yet.",
        [],
        llm,
    )
    computer = _agent(
        "computer",
        "Math Computer",
        "Execute each sub-step in order, using the calculator for arithmetic. Report the numeric result of each step.",
        [calculator],
        llm,
    )
    checker = _agent(
        "checker",
        "Math Checker",
        "Re-derive the final quantity via an alternative path to cross-check the computer. "
        "Do NOT emit the final \\boxed{...}.",
        [calculator],
        llm,
    )
    verifier = _agent(
        "verifier",
        "Math Verifier",
        "Reconcile the computer's and checker's results; emit the final answer inside \\boxed{...}.",
        [calculator],
        llm,
    )

    decompose_task = Task(
        description=(
            "Decompose the problem below into a short ordered list of "
            "computational sub-steps. Number each step. Do NOT compute "
            "the result — that is the next stage's job. Do NOT emit a "
            "final answer.\n\n"
            "PROBLEM:\n{problem}"
        ),
        expected_output="A numbered list of 3-8 computational sub-steps describing what the next stage must compute.",
        agent=decomposer,
    )
    compute_task = Task(
        description=(
            "Execute each sub-step from the Decomposer's list, in order. "
            "For each step, call the calculator tool on the numeric "
            "expression and report the result. End with your best "
            "current answer. Do NOT emit the final \\boxed{...} yet — "
            "that is the Verifier's job.\n\n"
            "PROBLEM:\n{problem}"
        ),
        expected_output=(
            "For each numbered step: the expression evaluated + the "
            "result. A final line stating your best-answer value "
            "derived from the last step."
        ),
        agent=computer,
        context=[decompose_task],
    )
    check_task = Task(
        description=(
            "Re-derive the final answer via an ALTERNATIVE path — either "
            "a symbolic simplification, a different decomposition, or a "
            "sanity-check identity. Use the calculator to verify your "
            "alternative. Report whether your result agrees with the "
            "Computer's; do NOT emit the final \\boxed{...} yet.\n\n"
            "PROBLEM:\n{problem}"
        ),
        expected_output=(
            "A short alternative derivation + a line stating 'Agrees "
            "with Computer' or 'Disagrees: Computer said X, Checker "
            "got Y'."
        ),
        agent=checker,
        context=[decompose_task, compute_task],
    )
    verify_task = Task(
        description=(
            "Reconcile the Computer's and Checker's results. If they "
            "agree, confirm. If they disagree, use the calculator to "
            "resolve (re-run the critical arithmetic). Your output "
            "MUST end with the final answer inside \\boxed{...}.\n\n"
            "PROBLEM:\n{problem}"
        ),
        expected_output="A short reconciliation note ending with the final boxed LaTeX answer (\\boxed{...}).",
        agent=verifier,
        context=[decompose_task, compute_task, check_task],
    )
    return Crew(
        agents=[decomposer, computer, checker, verifier],
        tasks=[decompose_task, compute_task, check_task, verify_task],
        process=Process.sequential,
        verbose=False,
    )


_STAGES = ("decomposer", "computer", "checker", "verifier")


def solve(problem: str) -> dict:
    """Run the crew on one problem.

    Returns ``{"answer", "raw", "by_stage", "telemetry"}``: the verifier's boxed
    answer and text, every stage's text and token/call counts.
    """
    crew = build_crew()
    result = crew.kickoff(inputs={"problem": problem})
    final = result.raw
    try:
        stages = {role: result.tasks_output[i].raw for i, role in enumerate(_STAGES)}
    except (AttributeError, IndexError):
        stages = {"decomposer": "", "computer": "", "checker": "", "verifier": final}
    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages,
        "telemetry": normalize(crewai_telemetry(crew, n_stages=len(stages))),
    }


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["problem"]))
        return task.record(
            inst,
            out["answer"],
            **task.meta(inst),
            by_stage=out.get("by_stage") or {},
            raw=out.get("raw") or "",
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label=f"sequential/{task.LABEL}")


def _canned_demo() -> None:
    out = solve(task.DEMO_PROBLEM)
    for role, text in out["by_stage"].items():
        print(f"\n=== {role.capitalize()} (excerpt) ===\n{text[:400]}...")
    task.print_demo_answer(out["answer"])


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Sequential-topology MATH runner (CrewAI 4-stage).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
