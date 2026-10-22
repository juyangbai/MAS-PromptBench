# Decentralized

Peer agents debate over several rounds with no coordinator. Each peer answers alone first, then revises after reading the other peers' previous answers, and the final round is put to a vote.
{ .lede }

--8<-- "diagrams/decentralized.svg"

<div class="facts" markdown>
<div><span>Agents</span>4 peers</div>
<div><span>Rounds</span>2</div>
<div><span>Frameworks</span>LangGraph, OpenAI Agents SDK</div>
<div><span>Aggregation</span>Final-round vote</div>
</div>

## How it works

The design follows multi-agent debate (Du et al. 2023, [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)).

1. Every peer gets the same `debater` seed prompt and the task.
2. **Round 0.** Each peer answers independently, using the dataset's tools.
3. **Round 1.** Each peer receives the other peers' final answers from round 0, with an instruction to revise only if a peer's reasoning or evidence is stronger. It then answers again.
4. Later rounds repeat step 3 with the previous round's answers. Peers inside a round run one after another, but each reads only the previous round, so no peer sees a same-round answer.
5. After the last round, the runner submits the majority of the peers' final answers, without looking at the gold answer or the tests:

| Datasets | Vote over |
| --- | --- |
| GPQA | extracted letters |
| HotpotQA | normalized short-form answers |
| MATH | buckets of `\boxed{}` answers that `is_equiv` treats as equal |
| BFCL | canonical function-call lists |
| LiveCodeBench, APPS | programs, compared with whitespace normalized; only the winner is tested |
| SWE-bench | non-empty patches, compared with whitespace normalized; only the winner is evaluated |
| ToolHop, API-Bank | final answers or API calls |

Ties go to the lowest peer index. The Agents SDK runners vote the same way over each peer's whole final output, compared with whitespace normalized.

From round 1 on, each peer reads the other `n - 1` peers' answers, so what it reads grows with team size.

The team spec sets 4 peers and 2 rounds (counting round 0); `DECENTRALIZED_N_AGENTS` and `DECENTRALIZED_N_ROUNDS` override them. The ToolHop and API-Bank runners read `TOOLHOP_`- or `APIBANK_`-prefixed versions of both first.

## Role and seed prompt

One role, `debater`, shared by every peer: `configs/prompts/decentralized/<dataset>/debater.txt` for all nine datasets. Both frameworks read the same file.

## Implementations

### LangGraph

`topologies/decentralized/langgraph/<dataset>/langgraph_<dataset>.py` builds a `StateGraph` with a single `round` node and a conditional edge that loops back to it until the last round is done. The state holds every peer's message history and every round's final answers. On most datasets each peer turn invokes one shared `create_react_agent` on that peer's history, and the other peers' answers arrive as one new user message. On SWE-bench, each peer edits its own clone of the repository.

### OpenAI Agents SDK

`topologies/decentralized/openai_agents/<dataset>/openai_agents_<dataset>.py` supplies the prompt, tools and scoring; the debate engine is [`agents_sdk_base.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/decentralized/openai_agents/agents_sdk_base.py) next to them. Every peer turn is one Agents SDK run with the dataset's function tools and no handoffs. From round 1 on, its input is the original task, the peer's own previous answer and the other peers' answers from the previous round. Each peer turn sends its own request seed.

These runners need the isolated SDK install from [Installation](../getting-started/installation.md#install-the-openai-agents-sdk); they restart themselves with it first on `PYTHONPATH`.

## Run it

=== "LangGraph"

    ```bash
    python -m topologies.decentralized.langgraph.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
      --out-dir results/topologies_baseline/decentralized_langgraph_hotpotqa
    ```

=== "OpenAI Agents SDK"

    ```bash
    python -m topologies.decentralized.openai_agents.hotpotqa.openai_agents_hotpotqa \
      --batch --limit 100 \
      --out-dir results/topologies_baseline/decentralized_openai_agents_hotpotqa
    ```

```bash title="Three rounds instead of two"
DECENTRALIZED_N_ROUNDS=3 python -m topologies.decentralized.langgraph.math.langgraph_math \
  --batch --limit 100 --out-dir results/decentralized_math_3rounds
```

See the [task pages](../tasks/index.md) for each dataset's options.

## Optimize it

| Protocol flags | What it runs |
| --- | --- |
| `--topology decentralized` | the LangGraph debate |
| `--topology decentralized --framework openai_agents` | the Agents SDK debate (`decentralized_openai_agents`) |
| `--topology decentralized --team-size N` | N peers (`decentralized_r<N>`), still 2 rounds |
| `--topology decentralized --communication FORMAT` | a [communication-protocol](communication-protocols.md) variant (`decentralized_communications_<format>`) |

The optimizer tunes the one `debater` prompt that every peer uses, and the job runs the same 4 peers and 2 rounds as the baseline. Any of the eight methods runs it; for example, HiveMind on BFCL:

```bash title="HiveMind on Decentralized · BFCL"
python -m optimizers.protocol.run --method hivemind --dataset bfcl --topology decentralized \
  --model qwen --seed 0 --out runs/hivemind/bfcl/decentralized/qwen/0
```

An Agents SDK job restarts itself with the SDK install first on `PYTHONPATH`, like the runners. See [Run an Optimizer](../optimizers/running.md).
