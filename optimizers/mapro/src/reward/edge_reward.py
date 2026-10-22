"""Edge reward g(p_i, p_j) = R_ij(y_i, D_j, P_j): an LLM judge scores the
handoff — whether upstream agent i's output y_i (under candidate p_i) enables
downstream agent j (operating under candidate p_j) to succeed.
"""

from __future__ import annotations

from ..llm import ChatModel, GenerationConfig
from .demos import DemoPool
from .node_reward import JUDGE_SYS, _clip
from .parse import parse_score

_EDGE_TMPL = """You judge the communication quality of a handoff from an upstream
agent to a downstream agent named '{dst_role}'.

Upstream agent's output (the message passed downstream):
{y_src}

The downstream agent will operate under this role prompt:
{dst_prompt}

Downstream preference demonstrations:
{demos}

Rate on a scale from 0.00 to 1.00 how well the upstream output enables the
downstream agent to perform its next step: information completeness, format,
clarity, and alignment with the downstream role. Return ONLY the floating-point
score."""

_JUDGE_CFG = GenerationConfig(temperature=0.2, max_tokens=8, enable_thinking=False)


async def edge_score(
    optimizer: ChatModel,
    dst_role: str,
    y_src: str,
    dst_prompt: str,
    demos: DemoPool,
    max_ctx: int = 2000,
) -> float:
    """Judged g(p_i, p_j): how well an upstream output serves a downstream candidate prompt."""
    prompt = _EDGE_TMPL.format(
        dst_role=dst_role,
        y_src=_clip(y_src, max_ctx),
        dst_prompt=_clip(dst_prompt, 3000),  # benchmark seeds run to ~2.5k chars
        demos=demos.render(),
    )
    out = await optimizer.chat_text(prompt, system=JUDGE_SYS, cfg=_JUDGE_CFG)
    return parse_score(out)


# --------------------------------------------------------------------------------
# Listwise edge judging (paper Fig. 4 edge_header + edge_reward_prefix; Eq. 5
# conditions on the WHOLE downstream pool P_j): one call per (upstream output,
# task) ranks all K_j downstream candidate prompts.
# --------------------------------------------------------------------------------
_EDGE_LIST_TMPL = """You are a *reward model* for assessing **communication quality** from an upstream agent to a downstream agent. Consider information completeness, format, clarity, and alignment with demonstrations.
Based on the input, output and prefernece examples,
you should first rank the candidate prompts with the good and bad examples,
Then you will give each a distinct two-decimal quality score between (0.00, 1.00) based on the standard and alignment with the good examples.
You should be severely harsh and the score difference should be ranged from 0.4 - 0.8 and each differs more than 0.05 with each other.
Finally, return exactly a score each line corresponding to the **prompt's original position**. (Not the sorted score)
Note that your output should contain only the numeric scores (e.g., 0.62). Nothing else.

You are an evaluation LLM. Judge whether the message produced by the upstream agent helps the downstream agent '{dst_role}' perform its next step, under each candidate downstream role prompt. Rate on a 0-1 scale. Use the demonstrations for guidance.

Upstream agent's output (the message passed downstream):
{y_src}

=== Preference Demonstrations ===
{demos}
=== End Demonstrations ===

Candidate downstream role prompts (in original position order):
{candidates}

Return exactly {k} lines: one two-decimal score per line, in the candidates' original order."""


async def edge_score_listwise(
    optimizer: ChatModel,
    dst_role: str,
    y_src: str,
    dst_prompts: list[str],
    demos: DemoPool,
    max_ctx: int = 2000,
) -> list[float]:
    """Judged edge scores of every downstream candidate in one ranked call."""
    from .node_reward import JUDGE_LIST_SYS
    from .parse import parse_score_list

    k = len(dst_prompts)
    candidates = "\n\n".join(f"[Candidate {i + 1}]\n{_clip(pr, 3000)}" for i, pr in enumerate(dst_prompts))
    prompt = _EDGE_LIST_TMPL.format(
        dst_role=dst_role,
        y_src=_clip(y_src, max_ctx),
        demos=demos.render(),
        candidates=candidates,
        k=k,
    )
    cfg = GenerationConfig(temperature=0.2, max_tokens=max(128, 40 * k), enable_thinking=False)
    out = await optimizer.chat_text(prompt, system=JUDGE_LIST_SYS, cfg=cfg)
    return parse_score_list(out, k)
