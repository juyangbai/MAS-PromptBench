"""Sequential GPQA-Diamond runner (CrewAI): analyzer -> solver -> critic -> verifier crew.

Each task sees the outputs of the earlier tasks as context; the answer is the
letter extracted from the verifier's output. The agents' backstories are the
sequential role prompts.
"""

from __future__ import annotations

from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from core import cli, prompts, settings
from core.batch import attempt
from core.calculator import CALCULATOR_DOC_DECIMAL_PI, make_calculator
from core.llm import crewai_llm
from core.tasks import gpqa as task
from core.tasks.gpqa import extract_answer, load_instances
from core.telemetry import crewai_telemetry, normalize

TOPOLOGY = "sequential"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

calculator = tool("calculator")(make_calculator(CALCULATOR_DOC_DECIMAL_PI))

format_mcq = task.format_prompt


def _load_prompt(role: str) -> str:
    return prompts.role_prompt(TOPOLOGY, task.DATASET, role)


def _build_llm() -> LLM:
    return crewai_llm(model=MODEL_ID, base_url=VLLM_BASE_URL, additional_drop_params=[])


def _agent(role: str, title: str, goal: str, model: LLM) -> Agent:
    return Agent(
        role=title,
        goal=goal,
        backstory=_load_prompt(role),
        tools=[calculator],
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
        "GPQA Analyzer",
        "Identify the scientific principles relevant to the MCQ and enumerate how each option would be derived.",
        llm,
    )
    solver = _agent(
        "solver",
        "GPQA Solver",
        "Apply the Analyzer's principles to choose one option and emit the final letter.",
        llm,
    )
    critic = _agent(
        "critic",
        "GPQA Critic",
        "Challenge the Solver's pick. For each rejected option, explain the strongest argument that WOULD "
        "defend it. Do NOT commit to a new final letter.",
        llm,
    )
    verifier = _agent(
        "verifier",
        "GPQA Verifier",
        "Reconcile the Solver's pick with the Critic's challenges; emit the final letter.",
        llm,
    )

    analyze_task = Task(
        description=(
            "Analyze the multiple-choice question below. Enumerate the "
            "scientific principles at play and describe, option by option "
            "(A, B, C, D), how each candidate answer would be derived. "
            "Do NOT commit to a final letter.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A structured analysis: a short 'Principles' section followed "
            "by one numbered paragraph per option (A/B/C/D) explaining "
            "each option's derivation."
        ),
        agent=analyzer,
    )
    solve_task = Task(
        description=(
            "Using the Analyzer's principles, select the single correct "
            "option and emit your reasoning + final letter. Your output "
            "MUST end with a line matching 'Answer: X' where X is one of "
            "A, B, C, D.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "Concise reasoning that applies the Analyzer's principles, followed by a final line of the form 'Answer: X'."
        ),
        agent=solver,
        context=[analyze_task],
    )
    critic_task = Task(
        description=(
            "Challenge the Solver's pick. Given the Analyzer's principles "
            "and the Solver's tentative letter + reasoning, identify any "
            "concrete errors in the Solver's logic and — for each "
            "REJECTED option — describe the strongest argument that "
            "would have defended it. Do NOT declare a new final letter.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A structured critique: 'Errors in solver's reasoning' "
            "section (may be empty if none), followed by 'Defense of "
            "rejected options' with a short note per option."
        ),
        agent=critic,
        context=[analyze_task, solve_task],
    )
    verify_task = Task(
        description=(
            "Reconcile the Solver's pick with the Critic's challenges. "
            "If the Critic exposed a concrete error, override; otherwise "
            "confirm. Your output MUST end with a line matching "
            "'Final answer: X' where X is one of A, B, C, D.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output="A short reconciliation note (confirm or override, with reason) ending with 'Final answer: X'.",
        agent=verifier,
        context=[analyze_task, solve_task, critic_task],
    )
    return Crew(
        agents=[analyzer, solver, critic, verifier],
        tasks=[analyze_task, solve_task, critic_task, verify_task],
        process=Process.sequential,
        verbose=False,
    )


_STAGES = ("analyzer", "solver", "critic", "verifier")


def solve(question: str, choices: list[str]) -> dict:
    """Run the crew on one question.

    Returns ``{"answer", "raw", "by_stage", "telemetry"}``: the verifier's
    letter and text, every stage's text and token/call counts.
    """
    crew = build_crew()
    result = crew.kickoff(inputs={"question": format_mcq(question, choices)})
    final = result.raw
    try:
        stages = {role: result.tasks_output[i].raw for i, role in enumerate(_STAGES)}
    except (AttributeError, IndexError):
        stages = {"analyzer": "", "solver": "", "critic": "", "verifier": final}
    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages,
        "telemetry": normalize(crewai_telemetry(crew, n_stages=len(stages))),
    }


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["question"], inst["choices"]))
        return task.record(
            inst,
            out["answer"],
            raw=out.get("raw") or "",
            by_stage=task.stage_excerpts(out.get("by_stage") or {}),
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label="sequential/GPQA-Diamond")


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION, task.DEMO_CHOICES)
    for role, text in out["by_stage"].items():
        print(f"\n=== {role.capitalize()} (excerpt) ===\n{text[:400]}...")
    task.print_demo_answer(out["answer"])


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Sequential-topology GPQA-Diamond runner (CrewAI).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
