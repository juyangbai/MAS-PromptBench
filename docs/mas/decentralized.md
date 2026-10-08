# Decentralized

Peer agents debate over several rounds with no coordinator. Each peer answers alone first, then revises after reading the other peers' previous answers, and the final round is put to a vote.
{ .lede }

--8<-- "diagrams/decentralized.svg"

<div class="facts" markdown>
<div><span>Agents</span>4 peers</div>
<div><span>Rounds</span>2</div>
<div><span>Frameworks</span>LangGraph, OpenAI SDK</div>
<div><span>Aggregation</span>Final-round vote</div>
</div>

## How it works

The design follows multi-agent debate (Du et al. 2023, [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)).

1. Every peer gets the same `debater` seed prompt and the task. Each peer keeps its own message history across rounds; in the ToolHop and API-Bank runners it gets its previous answer back as context instead.
2. **Round 0.** Each peer answers independently, using the dataset's tools.
3. **Round 1.** Each peer receives one new message holding the other peers' final answers from round 0, with an instruction to revise only if a peer's reasoning or evidence is stronger. It then answers again.
4. Later rounds repeat step 3 with the previous round's answers. Peers inside a round run one after another, but each reads only the previous round, so no peer sees a same-round answer.
5. After the last round, the runner aggregates the peers' final answers:

| Datasets | Rule |
| --- | --- |
| GPQA, HotpotQA, ToolHop, API-Bank | Majority over normalized answers; ties go to the lowest peer index. |
| MATH | Majority over buckets of equivalent `\boxed{}` answers. |
| LiveCodeBench, APPS | Run each program on the problem's tests; first that passes all, else the highest pass rate. |
| SWE-bench | First resolved patch, else the best fail-to-pass × pass-to-pass rate. |
| BFCL | First call the BFCL AST checker accepts, else the first peer with a call. |

From round 1 on, each peer reads the other `n - 1` peers' answers, so what it reads grows with team size.

Set the shape with `DECENTRALIZED_N_AGENTS` (default `4`) and `DECENTRALIZED_N_ROUNDS` (default `2`, counting round 0). The ToolHop and API-Bank runners read `TOOLHOP_`- or `APIBANK_`-prefixed versions of both first.

## Role and seed prompt

One role, `debater`, shared by every peer: `configs/prompts/decentralized/<dataset>/debater.txt` for all nine datasets. One exception: the OpenAI SDK runner for MATH reads `configs/prompts/decentralized_openai/math/debater.txt`, a separately generated prompt that also lists a `solve_equation` tool which only that runner provides.

## Implementations

### LangGraph

`topologies/decentralized/langgraph/<dataset>/langgraph_<dataset>.py` builds a `StateGraph` with a single `round` node and a conditional edge that loops back to it until `N_ROUNDS` rounds are done. The state holds every peer's message history and every round's final answers. On most datasets each peer turn invokes one shared `create_react_agent` on that peer's history.

### OpenAI SDK

`topologies/decentralized/openai/<dataset>/openai_<dataset>.py` uses the `openai` Python client directly: each peer turn is a `chat.completions.create` tool loop over a plain list of messages, and the peer message is appended as a user turn.

[`topologies/decentralized/openai/debate_base.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/decentralized/openai/debate_base.py) is the 70-line reference, adapted from `frameworks/llm_multiagent_debate`. Its demo defaults to 3 agents and 2 rounds; the benchmark runners use 4 and 2.

ToolHop and API-Bank runners use no framework objects. Both variants run the same plain-Python debate loop and differ only in their `STYLE` label.

## Run it

=== "LangGraph"

    ```bash
    python -m topologies.decentralized.langgraph.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
      --out results/topologies_baseline/decentralized_langgraph_hotpotqa/predictions.jsonl
    ```

=== "OpenAI SDK"

    ```bash
    python -m topologies.decentralized.openai.hotpotqa.openai_hotpotqa --batch --limit 100 \
      --out results/topologies_baseline/decentralized_openai_hotpotqa/predictions.jsonl
    ```

```bash title="Three rounds instead of two"
DECENTRALIZED_N_ROUNDS=3 python -m topologies.decentralized.langgraph.math.langgraph_math \
  --batch --limit 100 --out results/decentralized_math_3rounds/predictions.jsonl
```

BFCL, SWE-bench, ToolHop and API-Bank use `--out-dir`; see [Command-Line Flags](../reference/cli.md).

## Optimize it

| Optimizer topology name | What it runs |
| --- | --- |
| `decentralized` | the LangGraph debate |
| `decentralized_openai` | the OpenAI SDK debate |
| `decentralized_r2`, `decentralized_r4`, `decentralized_r8`, `decentralized_r10` | [team-size](team-sizes.md) variants (HotpotQA, LiveCodeBench, ToolHop, API-Bank) |
| `decentralized_communications_<format>` | [communication-protocol](communication-protocols.md) variants (same four datasets) |

The optimizer tunes the one `debater` prompt that every peer uses.

!!! warning "Set the debate shape on the optimizer"
    The optimizer pilots default to `--n-agents 2 --n-rounds 1`. With one round, peers never read each other. Pass `--n-agents 4 --n-rounds 2` to optimize the debate the baseline runs.

```bash title="GEPA on Decentralized / HotpotQA"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset hotpotqa --topology decentralized \
  --n-agents 4 --n-rounds 2 --train-size 25 --val-size 25 --max-full-evals 5 \
  --out results/gepa/decentralized_hotpotqa
```
