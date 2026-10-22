"""Node reward g(p_i) = R_i(x_i, y_i, D_i): an LLM judge scores how well agent
i's response y_i (produced under a candidate prompt) accomplishes its role,
given the input x_i and the agent's preference demonstrations D_i.
"""

from __future__ import annotations

from ..llm import ChatModel, GenerationConfig
from .demos import DemoPool
from .parse import parse_score

JUDGE_SYS = (
    "You are a strict reward model that evaluates the competence and clarity of a "
    "multi-agent-system role. You output ONLY a two-decimal floating-point number "
    "between 0.00 and 1.00."
)

_NODE_TMPL = """Agent role: {role}

Agent input:
{x}

Agent response (produced under the candidate role prompt):
{y}

Preference demonstrations (what good vs bad behavior looks like for this role):
{demos}

Rate on a scale from 0.00 to 1.00 how well the response accomplishes the agent's
role (correctness, completeness, clarity, and usefulness to downstream agents).
Return ONLY the floating-point score."""

_JUDGE_CFG = GenerationConfig(temperature=0.2, max_tokens=8, enable_thinking=False)


async def node_score(
    optimizer: ChatModel,
    role: str,
    x: str,
    y: str,
    demos: DemoPool,
    max_ctx: int = 2000,
) -> float:
    """Judged g(p_i): quality of one candidate's output for its role."""
    prompt = _NODE_TMPL.format(role=role, x=_clip(x, max_ctx), y=_clip(y, max_ctx), demos=demos.render())
    out = await optimizer.chat_text(prompt, system=JUDGE_SYS, cfg=_JUDGE_CFG)
    return parse_score(out)


def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"


# --------------------------------------------------------------------------------
# Listwise judging (paper Fig. 4, p.19 — verbatim node_header + agent_reward_prefix).
# Eq. 4 conditions the reward on the WHOLE pool P_i: one call ranks all K candidates
# with forced score separation, instead of K independent pointwise calls. Enabled by
# map_select/stage2_select when MAPRO_LISTWISE=1.
# --------------------------------------------------------------------------------
JUDGE_LIST_SYS = (
    "You are a strict reward model that evaluates the competence and clarity of a "
    "multi-agent-system role. You output ONLY numeric two-decimal scores, one per line."
)

_NODE_LIST_TMPL = """You are a *reward model* for evaluating the competence, clarity of candidate **role prompts**.
Based on the input, output and prefernece examples,
you should first rank the candidate prompts with the good and bad examples,
Then you will give each a distinct two-decimal quality score between (0.00, 1.00) based on the standard and alignment with the good examples.
You should be severely harsh and the score difference should be ranged from 0.4 - 0.8 and each differs more than 0.05 with each other.
Finally, return exactly a score each line corresponding to the **prompt's original position**. (Not the sorted score)
Note that your output should contain only the numeric scores (e.g., 0.62). Nothing else.

You are an evaluation LLM. Given the input and the agent's response, rate how well each response accomplishes the agent's role on a scale 0-1 (higher is better). Use the preference demonstrations below as reference.

Agent role: {role}

Agent input:
{x}

=== Preference Demonstrations ===
{demos}
=== End Demonstrations ===

Candidate role prompts and their resulting responses (in original position order):
{candidates}

Return exactly {k} lines: one two-decimal score per line, in the candidates' original order."""


async def node_score_listwise(
    optimizer: ChatModel,
    role: str,
    x: str,
    candidate_prompts: list[str],
    ys: list[str],
    demos: DemoPool,
    max_ctx: int = 2000,
) -> list[float]:
    """Judged node scores of every candidate in one ranked call."""
    from .parse import parse_score_list

    if len(candidate_prompts) != len(ys):
        raise ValueError("candidate_prompts and ys must have identical original-position order")
    k = len(candidate_prompts)
    cand_ctx = max_ctx  # full window per candidate (final answers sit at the tail)
    candidates = "\n\n".join(
        f"[Candidate {i + 1}]\nRole prompt:\n{_clip(candidate_prompt, 3000)}\nResponse:\n{_clip(y, cand_ctx)}"
        for i, (candidate_prompt, y) in enumerate(zip(candidate_prompts, ys))
    )
    prompt = _NODE_LIST_TMPL.format(role=role, x=_clip(x, max_ctx), demos=demos.render(), candidates=candidates, k=k)
    cfg = GenerationConfig(temperature=0.2, max_tokens=max(128, 40 * k), enable_thinking=False)
    out = await optimizer.chat_text(prompt, system=JUDGE_LIST_SYS, cfg=cfg)
    return parse_score_list(out, k)
