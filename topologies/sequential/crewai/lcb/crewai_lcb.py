"""Sequential LCB runner (CrewAI): analyzer -> coder -> tester -> debugger crew.

Each task sees the outputs of the earlier tasks as context; the submission is the
debugger's last fenced Python block (else the coder's). The agents' backstories
are the sequential role prompts.
"""

from __future__ import annotations

from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from core import cli, code_tasks, prompts, settings
from core.batch import attempt
from core.code_tasks import exact_match_score, extract_code  # noqa: F401  (runner API)
from core.llm import crewai_llm
from core.tasks import lcb as task
from core.tasks.lcb import format_prompt, load_instances, run_tests  # noqa: F401
from core.telemetry import crewai_telemetry, normalize

TOPOLOGY = "sequential"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

python_exec = tool("python_exec")(code_tasks.make_python_exec(code_tasks.PYTHON_EXEC_DOC))


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
    analyzer = _agent(
        "analyzer",
        "LCB Algorithm Analyzer",
        "Read the problem and choose a correct algorithmic approach: data structures, complexity, edge cases. "
        "Do NOT write code.",
        [],
        llm,
    )
    coder = _agent(
        "coder",
        "LCB Coder",
        "Implement the Analyzer's approach in Python, following the required I/O format (stdin or starter_code "
        "call-based).",
        [],
        llm,
    )
    tester = _agent(
        "tester",
        "LCB Tester",
        "Construct 3-5 targeted test cases (including edge cases) and run the Coder's program against them via "
        "python_exec; report which pass or fail with concrete evidence. Do NOT modify the Coder's program.",
        [python_exec],
        llm,
    )
    debugger = _agent(
        "debugger",
        "LCB Debugger",
        "Fix any failures the Tester reported. If all tests passed, pass the Coder's program through. Emit the "
        "final fenced Python block.",
        [python_exec],
        llm,
    )

    analyze_task = Task(
        description=(
            "Read the programming problem below. Pick a correct "
            "algorithmic approach: name the data structures, target "
            "complexity, and the edge cases that will bite a naive "
            "implementation. Do NOT write code — the Coder handles that.\n\n"
            "PROBLEM:\n{problem_prompt}"
        ),
        expected_output=(
            "A short plan: (1) approach summary, (2) chosen data "
            "structures, (3) target time/space complexity, (4) edge "
            "cases to handle."
        ),
        agent=analyzer,
    )
    code_task = Task(
        description=(
            "Implement the Analyzer's plan in Python. Follow the I/O "
            "format specified in the problem (stdin or starter_code "
            "call-based). Emit the complete Python program inside a "
            "fenced ```python ... ``` block.\n\n"
            "PROBLEM:\n{problem_prompt}"
        ),
        expected_output="A complete Python program inside a ```python ... ``` block.",
        agent=coder,
        context=[analyze_task],
    )
    test_task = Task(
        description=(
            "Construct 3-5 targeted test cases for the Coder's program, "
            "including at least one edge case. Run each case via "
            "python_exec. Report which cases pass or fail with the "
            "stdout/stderr evidence. Do NOT modify the program.\n\n"
            "PROBLEM:\n{problem_prompt}"
        ),
        expected_output=(
            "A list of test cases and per-case verdict (pass/fail); for failures include expected vs actual output."
        ),
        agent=tester,
        context=[analyze_task, code_task],
    )
    debug_task = Task(
        description=(
            "If the Tester reported failures, fix the Coder's program "
            "to address them (use python_exec to verify your fix). If "
            "the Tester reported all-pass, pass the Coder's program "
            "through unchanged. Your final output MUST contain the "
            "final Python program inside a fenced ```python ... ``` "
            "block (the extractor takes the last one).\n\n"
            "PROBLEM:\n{problem_prompt}"
        ),
        expected_output=(
            "A short note on what (if anything) was fixed + the final "
            "Python program inside a fenced ```python ... ``` block."
        ),
        agent=debugger,
        context=[analyze_task, code_task, test_task],
    )
    return Crew(
        agents=[analyzer, coder, tester, debugger],
        tasks=[analyze_task, code_task, test_task, debug_task],
        process=Process.sequential,
        verbose=False,
    )


_STAGES = ("analyzer", "coder", "tester", "debugger")


def solve(problem: str, starter_code: str | None = None) -> dict:
    """Run the crew on one problem (functional mode with ``starter_code``, stdin mode without).

    Returns ``{"code", "raw", "by_stage", "telemetry"}``: the debugger's program (else
    the coder's), the crew's final text, every stage's text and token/call counts.
    """
    crew = build_crew()
    result = crew.kickoff(inputs={"problem_prompt": format_prompt(problem, starter_code)})
    final = result.raw
    try:
        stages = {role: result.tasks_output[i].raw for i, role in enumerate(_STAGES)}
    except (AttributeError, IndexError):
        stages = {"analyzer": "", "coder": "", "tester": "", "debugger": final}
    code = extract_code(stages.get("debugger", "") or final)
    if code is None:
        code = extract_code(stages.get("coder", ""))
    return {
        "code": code,
        "raw": final,
        "by_stage": stages,
        "telemetry": normalize(crewai_telemetry(crew, n_stages=len(stages))),
    }


def run_batch(
    instances: list[dict],
    out_path: Path | None = None,
    verbose: bool = True,
    per_test_timeout_s: int = task.BATCH_TEST_TIMEOUT_S,
) -> dict:
    """Solve every instance and score it on its tests."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(
            lambda: solve(inst["problem"], starter_code=inst.get("starter_code") or None), fallback={"code": None}
        )
        code = out["code"]
        return task.record(
            inst,
            code,
            task.test_scores(code, inst["tests"], per_test_timeout_s),
            by_stage=out.get("by_stage") or {},
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label="sequential/LCB")


def _canned_demo() -> None:
    _, problem, _, tests = task.DEMOS[0]
    out = solve(problem)
    for role, text in out["by_stage"].items():
        print(f"\n=== {role.capitalize()} (excerpt) ===\n{text[:400]}...")
    task.print_demo_code(out["code"], tests)


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Sequential-topology LCB runner (CrewAI 4-stage).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
        add_arguments=task.add_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
