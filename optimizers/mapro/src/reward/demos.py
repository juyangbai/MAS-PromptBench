"""Preference demonstration pools D_i and the critic that maintains them.

A demonstration is a short (prompt, output, note) record labelled '+' (accepted)
or '-' (rejected). The reward judge conditions on ≤3 of these to calibrate its
scoring toward what "good" looks like for a given agent role (paper: preference
demonstrations, capped at 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class Demo:
    """One labelled ('+' accepted / '-' rejected) prompt-output demonstration."""

    prompt: str  # the role prompt that produced the behavior
    output: str  # an example agent output (may be truncated)
    label: str  # "+" or "-"
    note: str = ""  # short reason it is good/bad

    def render(self, max_out: int = 400) -> str:
        """One-line rendering for a judge prompt."""
        out = self.output.strip().replace("\n", " ")
        if len(out) > max_out:
            out = out[:max_out] + "…"
        tag = "GOOD" if self.label == "+" else "BAD"
        line = f"[{tag}] output: {out}"
        if self.note:
            line += f"  (why: {self.note.strip()})"
        return line


@dataclass
class DemoPool:
    """Positive and negative demonstrations of one agent role."""

    positives: list[Demo] = field(default_factory=list)
    negatives: list[Demo] = field(default_factory=list)
    max_demos: int = 3  # total, per paper

    def add(self, demo: Demo) -> None:
        """File a demonstration under its polarity."""
        (self.positives if demo.label == "+" else self.negatives).append(demo)

    def render(self) -> str:
        """Format ≤max_demos demos (balanced pos/neg, most recent first)."""
        if not self.positives and not self.negatives:
            return "(no demonstrations yet)"
        chosen: list[Demo] = []
        pos = list(reversed(self.positives))
        neg = list(reversed(self.negatives))
        while len(chosen) < self.max_demos and (pos or neg):
            if pos and (len(chosen) % 2 == 0 or not neg):
                chosen.append(pos.pop(0))
            elif neg:
                chosen.append(neg.pop(0))
        return "\n".join(d.render() for d in chosen)

    def trim(self, keep: int = 6) -> None:
        """Bound memory: keep only the most recent `keep` of each polarity."""
        self.positives = self.positives[-keep:]
        self.negatives = self.negatives[-keep:]


def critic_update(
    pool: DemoPool,
    chosen_prompt: str,
    chosen_output: str,
    node_scores: Sequence[float],  # (K,) array of g(p_i^k) for this agent
    candidate_prompts: list[str],
    candidate_outputs: list[str],
    success: bool,
    score_hi: float = 0.75,
    score_lo: float = 0.4,
) -> None:
    """Critic: D_i ← Critic(D_i, P_i, g(P_i)).

    Reclassify demonstrations from the round's reward scores + actual performance:
      * the chosen prompt's output becomes a POSITIVE demo iff the run succeeded
        and its node score is high;
      * the lowest-scoring candidate (below score_lo) becomes a NEGATIVE demo.
    This is a score-driven reclassification — the judge output *is* g(P_i), so no
    extra LLM call is needed.
    """
    import numpy as np

    scores = np.asarray(node_scores, dtype=float)
    ci = candidate_prompts.index(chosen_prompt) if chosen_prompt in candidate_prompts else int(scores.argmax())

    if success and scores[ci] >= score_hi:
        pool.add(Demo(prompt=chosen_prompt, output=chosen_output, label="+", note="selected prompt on a solved task"))

    worst = int(scores.argmin())
    if scores[worst] <= score_lo and worst != ci:
        pool.add(
            Demo(
                prompt=candidate_prompts[worst],
                output=candidate_outputs[worst],
                label="-",
                note=f"low reward score {scores[worst]:.2f}",
            )
        )

    pool.trim()
