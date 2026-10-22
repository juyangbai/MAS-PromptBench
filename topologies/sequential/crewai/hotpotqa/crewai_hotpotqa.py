"""Sequential HotpotQA runner (CrewAI): planner -> retriever -> reasoner -> writer crew.

Each task sees the outputs of the earlier tasks as context; only the retriever
has the Wikipedia tools. The answer is the writer's ``Answer:`` line. The agents'
backstories are the sequential role prompts.
"""

from __future__ import annotations

from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from core import cli, prompts, settings
from core.batch import attempt
from core.llm import crewai_llm
from core.tasks import hotpotqa as task
from core.tasks.hotpotqa import (  # noqa: F401  (runner API)
    exact_match_score,
    extract_answer,
    f1_score,
    load_instances,
    normalize_answer,
)
from core.telemetry import crewai_telemetry, normalize

TOPOLOGY = "sequential"

VLLM_BASE_URL = settings.base_url()
MODEL_ID = settings.model_id()

wikipedia_search = tool("wikipedia_search")(task.make_wikipedia_search(task.SEARCH_DOC))
wikipedia_page = tool("wikipedia_page")(task.make_wikipedia_page(task.PAGE_DOC))


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
    planner = _agent(
        "planner",
        "HotpotQA Planner",
        "Identify the entities in the question and plan the 2-3 hop Wikipedia search strategy "
        "BEFORE any retrieval happens. Do NOT retrieve.",
        [],
        llm,
    )
    retriever = _agent(
        "retriever",
        "HotpotQA Retriever",
        "Execute the Planner's queries via wikipedia_search and wikipedia_page; return a structured fact dossier.",
        [wikipedia_search, wikipedia_page],
        llm,
    )
    reasoner = _agent(
        "reasoner",
        "HotpotQA Reasoner",
        "Chain facts from the retrieved articles into a logical path to the answer.",
        [],
        llm,
    )
    writer = _agent("writer", "HotpotQA Writer", "Emit the concise final short-form answer.", [], llm)

    plan_task = Task(
        description=(
            "Read the HotpotQA question and plan the search strategy. "
            "Identify the key entities, decide which Wikipedia pages "
            "should be consulted and in what order, and what facts from "
            "each page would settle the question. Do NOT retrieve — the "
            "Retriever handles that.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A short plan: (1) key entities, (2) ordered list of "
            "Wikipedia searches to perform, (3) what fact each search "
            "should return."
        ),
        agent=planner,
    )
    retrieve_task = Task(
        description=(
            "Execute the Planner's search plan via `wikipedia_search` "
            "and `wikipedia_page`. Extract verbatim the facts the "
            "Planner flagged as load-bearing. Do NOT commit to a final "
            "answer.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output=(
            "A structured dossier: a list of the Wikipedia article titles "
            "consulted and, under each title, 1-3 verbatim facts that "
            "bear on the question."
        ),
        agent=retriever,
        context=[plan_task],
    )
    reason_task = Task(
        description=(
            "Using the Retriever's dossier, reason step by step from the "
            "facts to the answer. Make the logical chain explicit: "
            "'Fact A says X; Fact B says Y; therefore ...'. Do NOT emit "
            "the final answer string yet.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output="A short chain-of-reasoning paragraph that derives the answer from the Retriever's facts.",
        agent=reasoner,
        context=[plan_task, retrieve_task],
    )
    write_task = Task(
        description=(
            "Given the Reasoner's derivation, emit the final short-form "
            "answer. HotpotQA answers are typically 1-5 words "
            "(an entity name, a year, a yes/no, a number). Your output "
            "MUST end with a line matching 'Answer: <short form>'.\n\n"
            "QUESTION:\n{question}"
        ),
        expected_output="A one-line answer formatted as 'Answer: <short form>'.",
        agent=writer,
        context=[plan_task, retrieve_task, reason_task],
    )
    return Crew(
        agents=[planner, retriever, reasoner, writer],
        tasks=[plan_task, retrieve_task, reason_task, write_task],
        process=Process.sequential,
        verbose=False,
    )


_STAGES = ("planner", "retriever", "reasoner", "writer")


def solve(question: str) -> dict:
    """Run the crew on one question.

    Returns ``{"answer", "raw", "by_stage", "telemetry"}``: the writer's
    short-form answer and text, every stage's text and token/call counts.
    """
    crew = build_crew()
    result = crew.kickoff(inputs={"question": question})
    final = result.raw
    try:
        stages = {role: result.tasks_output[i].raw for i, role in enumerate(_STAGES)}
    except (AttributeError, IndexError):
        stages = {"planner": "", "retriever": "", "reasoner": "", "writer": final}
    return {
        "answer": extract_answer(final),
        "raw": final,
        "by_stage": stages,
        "telemetry": normalize(crewai_telemetry(crew, n_stages=len(stages))),
    }


def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True) -> dict:
    """Solve and score every instance; records keep the first 800 characters of each stage."""

    def row(_, inst: dict) -> dict:
        out, latency_s, error = attempt(lambda: solve(inst["question"]))
        return task.record(
            inst,
            out["answer"],
            **task.meta(inst),
            raw=out.get("raw") or "",
            by_stage={role: (text or "")[:800] for role, text in (out.get("by_stage") or {}).items()},
            latency_s=round(latency_s, 2),
            **(out.get("telemetry") or {}),
            error=error,
        )

    return task.run_batch(instances, row, out_path=out_path, verbose=verbose, label="sequential/HotpotQA")


def _canned_demo() -> None:
    out = solve(task.DEMO_QUESTION)
    for role, text in out["by_stage"].items():
        print(f"\n=== {role.capitalize()} (excerpt) ===\n{text[:400]}...")
    task.print_demo_answer(out["answer"])


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        description="Sequential-topology HotpotQA runner (CrewAI).",
        load_instances=load_instances,
        run_batch=run_batch,
        demo=_canned_demo,
        source=task.SOURCE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
