"""Prompt-pool construction and mutation.

Stage 1 (init): build a pool of K candidates by generating semantically-similar
variants of an agent's base prompt.

Stage 3 (refine): from the current best prompt p_i* plus execution feedback
(global f_g + local blame f_l), produce K-1 controlled variants via three
operators (a trust-region edit set), always keeping p_i* itself:
    * Adding        — append one concrete sentence from the feedback.
    * Replacement   — rewrite the weak part(s) with better instructions.
    * Reorganization— clean/reorganize for logical consistency, intent preserved.
"""

from __future__ import annotations

import asyncio
import re

from ...regime import MAPROSettings
from ..llm import ChatModel, GenerationConfig

_REWRITER_SYS = (
    "You rewrite role prompts for a single agent inside a multi-agent system. "
    "You output ONLY the rewritten role prompt text, with no preamble, quotes, or labels."
)

_INIT_TMPL = """Agent role: {role}

Base role prompt:
\"\"\"{base}\"\"\"

Produce {n} alternative versions of this role prompt. Each must preserve the
role's intent but vary the wording, emphasis, or structure. Keep each concise
(1-4 sentences). Return each variant on its own line prefixed with 'VARIANT:'."""

MUTATION_INSTRUCTIONS = {
    "adding": (
        "Apply the ADDING operator: keep the prompt as-is but append exactly one "
        "concrete sentence, drawn from the feedback, that would fix the observed problem."
    ),
    "replacement": (
        "Apply the REPLACEMENT operator: rewrite the weak or error-prone part(s) of the "
        "prompt with clearer, more specific instructions that address the feedback."
    ),
    "reorganization": (
        "Apply the REORGANIZATION operator: re-organize and clean the prompt so it is "
        "logically consistent and easy to follow, without changing its intent."
    ),
}
_OPERATORS = list(MUTATION_INSTRUCTIONS)

_MUTATE_TMPL = """Agent role: {role}

Current role prompt:
\"\"\"{prompt}\"\"\"

Execution feedback for this agent:
- Global outcome: {f_g}
- Targeted critique (blame from downstream): {f_l}

{operator}

(Revision attempt {nonce} — produce a distinct improvement.)

Return ONLY the new role prompt text."""

# Native output caps (``MAPROSettings``: MAPRO_INIT_MAX_TOKENS / MAPRO_MUTATE_MAX_TOKENS).
# The protocol integration sends every rewriter call with the common reflection
# ceiling, so these caps only apply when the functions are driven by a plain client.
_ENV = MAPROSettings.from_env()
_INIT_CFG = GenerationConfig(temperature=0.7, max_tokens=_ENV.init_max_tokens, enable_thinking=False)
_MUTATE_CFG = GenerationConfig(temperature=0.7, max_tokens=_ENV.mutate_max_tokens, enable_thinking=False)


def _clean(text: str) -> str:
    t = text.strip()
    t = re.sub(
        r"^\W*(?:\d{1,2}[.)]\s*)?\**\s*(VARIANT|PROMPT|ROLE PROMPT)\s*\d*\s*\**\s*[:.\-\u2013]\s*\**\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return t.strip().strip('"').strip()


_VARIANT_MARK = re.compile(r"^\W*(?:\d{1,2}[.)]\s*)?\**\s*VARIANT\s*\d*\s*\**\s*[:.\-\u2013]\s*\**", re.I | re.M)


def _split_variants(out: str) -> list[str]:
    """Split a rewriter reply on VARIANT markers: the text after each marker up to the next
    one is one candidate (works for "VARIANT 1: text" lines AND "VARIANT 1:\n<text>" blocks;
    the old line filter turned the latter into a pool of preamble + base copies)."""
    marks = list(_VARIANT_MARK.finditer(out))
    variants = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(out)
        text = out[m.end() : end].strip().strip('"').strip()
        if text:
            variants.append(text)
    return variants


async def init_candidates(optimizer: ChatModel, role: str, base_prompt: str, K: int) -> list[str]:
    """Return a pool of K candidates: base_prompt + (K-1) reworded variants."""
    if K <= 1:
        return [base_prompt]
    out = await optimizer.chat_text(
        _INIT_TMPL.format(role=role, base=base_prompt, n=K - 1),
        system=_REWRITER_SYS,
        cfg=_INIT_CFG,
    )
    variants = _split_variants(out)  # handles "VARIANT 1:\n<text>" block layouts too
    if len(variants) < K - 1:  # legacy one-variant-per-line format
        variants = [_clean(l) for l in out.splitlines() if "VARIANT" in l.upper()]
    if len(variants) < K - 1:  # fallback: any non-empty lines
        variants = [_clean(l) for l in out.splitlines() if _clean(l)]
    variants = [v for v in variants if v][: K - 1]
    while len(variants) < K - 1:  # pad with base if the model under-produced
        variants.append(base_prompt)
    return [base_prompt] + variants


async def _mutate_one(
    optimizer: ChatModel,
    role: str,
    prompt: str,
    f_g: str,
    f_l: str,
    operator: str,
    nonce: str,
) -> str:
    out = await optimizer.chat_text(
        _MUTATE_TMPL.format(
            role=role,
            prompt=prompt,
            f_g=f_g or "(none)",
            f_l=f_l or "(none)",
            operator=MUTATION_INSTRUCTIONS[operator],
            nonce=nonce,
        ),
        system=_REWRITER_SYS,
        cfg=_MUTATE_CFG,
    )
    cleaned = _clean(out)
    return cleaned or prompt


async def mutate_pool(
    optimizer: ChatModel,
    role: str,
    best_prompt: str,
    f_g: str,
    f_l: str,
    K: int,
    nonce: str = "",
) -> list[str]:
    """New pool = { p* } ∪ { K-1 controlled mutations of p* }.

    `nonce` (e.g. the iteration index) diversifies mutations across rounds so the
    response cache does not stall exploration when the incumbent is unchanged.
    """
    if K <= 1:
        return [best_prompt]
    jobs = [
        _mutate_one(optimizer, role, best_prompt, f_g, f_l, _OPERATORS[m % len(_OPERATORS)], f"{nonce}.{m}")
        for m in range(K - 1)
    ]
    variants = await asyncio.gather(*jobs)
    return [best_prompt] + [v for v in variants]
