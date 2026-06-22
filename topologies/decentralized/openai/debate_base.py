"""Decentralized debate topology (Du et al. 2023, arXiv:2305.14325).

N agents answer the same prompt over R rounds. Round 0 is independent;
in every later round each agent sees the previous round's answers from
all other agents (complete-graph peer-to-peer, no central coordinator).

Adapted from frameworks/llm_multiagent_debate/gsm/gen_gsm.py and
modernized for openai>=2.
"""

import os

from openai import OpenAI


def construct_peer_message(others: list[list[dict]], prompt: str, idx: int) -> dict:
    """Build the user-turn that injects peers' last-round answers."""
    if not others:
        return {
            "role": "user",
            "content": "Double-check and reiterate your answer.",
        }

    body = "These are responses from other agents:\n"
    for ctx in others:
        body += f"\n One agent response: ```{ctx[idx]['content']}```\n"
    body += (
        "\n Using the responses from other agents as additional information, "
        "provide your updated answer to the original prompt:\n"
        f"{prompt}"
    )
    return {"role": "user", "content": body}


def run_debate(
    prompt: str,
    n_agents: int = 3,
    n_rounds: int = 2,
    model: str = "gpt-4o-mini",
) -> list[list[dict]]:
    client = OpenAI()
    contexts = [[{"role": "user", "content": prompt}] for _ in range(n_agents)]

    for r in range(n_rounds):
        for i, ctx in enumerate(contexts):
            if r != 0:
                others = contexts[:i] + contexts[i + 1 :]
                # 2*r - 1 indexes the assistant reply from round r-1, avoiding
                # leakage of in-progress round-r replies from earlier agents.
                ctx.append(construct_peer_message(others, prompt, 2 * r - 1))

            resp = client.chat.completions.create(model=model, messages=ctx, n=1)
            ctx.append({"role": "assistant", "content": resp.choices[0].message.content})

    return contexts


if __name__ == "__main__":
    assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY"

    prompt = (
        "Solve: a train leaves city A at 60 mph and another leaves city B "
        "at 80 mph toward each other. They are 420 miles apart. When do they meet? "
        "Give the answer in hours, in the form \\boxed{answer}."
    )
    contexts = run_debate(prompt, n_agents=3, n_rounds=2)

    for i, ctx in enumerate(contexts):
        print(f"\n=== Agent {i} (final answer) ===")
        print(ctx[-1]["content"])
