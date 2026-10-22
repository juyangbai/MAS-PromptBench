# All Tasks

MAS-PromptBench runs every topology and optimizer on nine existing benchmarks, three each for reasoning, coding and tool calling. This page lists what each task asks for, how it is scored and what it needs before you run it.
{ .lede }

## Three domains

The tasks are grouped by what the final agent has to produce and how that output is checked:

- **Reasoning**: GPQA-Diamond, HotpotQA and MATH. The agents end with a short answer (a letter, a short phrase, a boxed expression), and the scorer compares it with a gold answer after normalization.
- **Coding**: LiveCodeBench, APPS and SWE-bench Verified. The agents produce a program or a repository patch, and the scorer runs tests against it.
- **Tool calling**: BFCL, ToolHop and API-Bank. The agents emit function or API calls against schemas given with each task, and the scorer checks the calls themselves or the answer the call chain produces.

Each task fixes the form of the final answer with an output contract, defined in [`core/output_contracts.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/core/output_contracts.py). The runner adds it to the answering role's system prompt at load time, so an optimizer can rewrite role prompts without removing the format the scorer expects. Data loading, scoring and records live in one task module per dataset, `core/tasks/<dataset>.py`.

## At a glance

| Task | CLI name | Domain | Metric (scorer) | Eval IDs | Train / val | Special setup |
| --- | --- | --- | --- | ---: | ---: | --- |
| [GPQA-Diamond](gpqa.md) | `gpqa` | Reasoning | Accuracy (letter match) | 100 | 48 / 50 | Gated Hugging Face dataset |
| [HotpotQA](hotpotqa.md) | `hotpotqa` | Reasoning | Exact match and F1 (official) | 100 | 150 / 50 | Network access to Wikipedia |
| [MATH](math.md) | `math` | Reasoning | Accuracy (Hendrycks `is_equiv`) | 100 | 150 / 50 | None |
| [LiveCodeBench](livecodebench.md) | `lcb` | Coding | pass@1 | 50 | 150 / 50 | Runs generated code on the host |
| [APPS](apps.md) | `apps` | Coding | Strict accuracy (pass@1) | 50 | 150 / 50 | Runs generated code on the host |
| [SWE-bench Verified](swe-bench.md) | `swe` | Coding | Resolved rate | 30 | 146 / 50 | git, Singularity, per-instance images |
| [BFCL](bfcl.md) | `bfcl` | Tool calling | AST match (`bfcl-eval`) | 100 | 150 / 50 | None |
| [ToolHop](toolhop.md) | `toolhop` | Tool calling | Answer accuracy | 100 | 150 / 50 | `TOOLHOP_ALLOW_DATASET_EXEC=1` |
| [API-Bank](api-bank.md) | `apibank` | Tool calling | API-call accuracy (API-Bank checkers) | 100 | 150 / 50 | None (source ships in the repo) |

The CLI name is the task's folder name and the optimizers' `--dataset` value. The eval IDs, 730 in total, are the instances behind the reported scores, listed in `benchmarks/<task>/<task>_eval_ids.json`. They are also the `test` split of `benchmarks/<task>/<task>_splits.json`; the `train` and `validation` splits the optimizers use are drawn from the rest, so optimization never sees a reported instance.

<div class="cards" markdown>

- [GPQA-Diamond](gpqa.md)
  Graduate-level science questions with four options.
- [HotpotQA](hotpotqa.md)
  Multi-hop questions answered from live Wikipedia.
- [MATH](math.md)
  Level 5 precalculus problems with a boxed final answer.
- [LiveCodeBench](livecodebench.md)
  Contest programming problems judged on hidden tests.
- [APPS](apps.md)
  Python programming problems judged on capped test sets.
- [SWE-bench Verified](swe-bench.md)
  Real GitHub issues fixed by editing the repository.
- [BFCL](bfcl.md)
  Function calls checked by the official AST checker.
- [ToolHop](toolhop.md)
  Chains of tool calls that end in one exact answer.
- [API-Bank](api-bank.md)
  The next API call in a dialogue, executed and checked.

</div>

## Running a task

Every task has one runner per topology variant, at `topologies/<topology>/[<framework>/]<task>/<framework>_<task>.py`. Run them from the repository root as modules:

```bash title="Run a runner as a module"
python -m topologies.single.gpqa.langgraph_gpqa --batch --limit 100 \
  --out-dir results/topologies_baseline/single_gpqa
```

All runners share one command line, plus a few options per dataset:

| Flag | Effect |
| --- | --- |
| `--batch` | Run the dataset instead of the canned demo. |
| `--limit N` | Evaluate at most N instances, after `--offset`. |
| `--offset K` | Skip the first K instances. |
| `--only ID ...` | Evaluate only these IDs, whatever `--limit` says; repeat the flag or list several IDs after it. |
| `--out-dir DIR` | Write `DIR/predictions.jsonl` and the runner's other files there. |
| `--out PATH` | Write the predictions to PATH instead. |

Without `--batch`, the GPQA, HotpotQA, MATH, LiveCodeBench and APPS runners play a built-in demo; the BFCL, SWE-bench, ToolHop and API-Bank runners have none and run a small batch (5, 2, 5 and 2 instances). Every output file is emptied when a batch starts. Without `--out-dir` or `--out`, some runners write to their own folder under `results/` and others only print their scores, so pass one. Progress goes to stderr; set `LOG_LEVEL` to change how much. [Command-Line Flags](../reference/cli.md) lists every option.

!!! tip "Score the evaluation IDs"
    For seven tasks the first instances are exactly the eval IDs, so `--limit` with the eval-ID count scores the reported set. BFCL and SWE-bench are samples: pass their IDs to `--only`, as their pages show. `scripts/run_topologies.sh` runs every topology on every task this way.

## Optimize a task

Every optimizer runs a task through one command, with the task's CLI name as `--dataset`:

```bash title="One optimizer job"
python -m optimizers.protocol.run --method <key> --dataset <task> --topology <topology> \
  --model qwen --seed 0 --out runs/<key>/<task>/<topology>/qwen/0
```

`<key>` is one of `gepa`, `mipro`, `mapro`, `maspo`, `hivemind`, `mamut_gepa`, `maspob` or `tavo`. The experiment grid pairs GEPA, MIPRO, MAPRO and MASPO with every task and every topology runner, in each of its frameworks (on Single, GEPA and MIPRO only), and all eight methods with the multi-agent LangGraph teams of HotpotQA, LiveCodeBench and BFCL. Other combinations need `--allow-any-cell` and are reported as non-conformant.

During optimization, every rollout is scored 0 or 1 by the task's protocol metric. Most tasks use the runner's own scorer; HotpotQA uses exact match only, LiveCodeBench and APPS run only the first 3 tests, and SWE-bench uses a structural patch check. See [Run an Optimizer](../optimizers/running.md) and [Evaluation Protocol](../evaluation/protocol.md).
