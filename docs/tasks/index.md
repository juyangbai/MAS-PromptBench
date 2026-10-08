# All Tasks

MAS-PromptBench runs every topology and optimizer on nine existing benchmarks, three each for reasoning, coding and tool calling. This page lists what each task asks for, how it is scored and what it needs before you run it.
{ .lede }

## Three domains

The tasks are grouped by what the final agent has to produce and how that output is checked:

- **Reasoning**: GPQA-Diamond, HotpotQA and MATH. The agents end with a short answer (a letter, a short phrase, a boxed expression), and the scorer compares it with a gold answer after normalization.
- **Coding**: LiveCodeBench, APPS and SWE-bench Verified. The agents produce a program or a repository patch, and the scorer runs tests against it.
- **Tool calling**: BFCL, ToolHop and API-Bank. The agents emit function or API calls against schemas given with each task, and the scorer checks the calls themselves or the answer the call chain produces.

Each task fixes the form of the final answer with an output contract, defined in [`topologies/output_contracts.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/output_contracts.py). The runner adds it to the final role's system prompt at load time, so an optimizer can rewrite role prompts without removing the format the scorer expects. The paper also reports its results per domain.

## At a glance

| Task | CLI name | Domain | Metric (scorer) | Launcher limit | Frozen IDs | Special setup |
| --- | --- | --- | --- | ---: | ---: | --- |
| [GPQA-Diamond](gpqa.md) | `gpqa` | Reasoning | Accuracy (letter match) | 100 | 100 | Gated Hugging Face dataset |
| [HotpotQA](hotpotqa.md) | `hotpotqa` | Reasoning | Exact match and F1 (official) | 100 | 100 | Network access to Wikipedia |
| [MATH](math.md) | `math` | Reasoning | Accuracy (Hendrycks `is_equiv`) | 100 | 100 | None |
| [LiveCodeBench](livecodebench.md) | `lcb` | Coding | pass@1 | 50 | 50 | Runs generated code on the host |
| [APPS](apps.md) | `apps` | Coding | Strict accuracy (pass@1) | 50 | 50 | Runs generated code on the host |
| [SWE-bench Verified](swe-bench.md) | `swe` | Coding | Resolved rate | 30 | 30 | git, Singularity, per-instance images |
| [BFCL](bfcl.md) | `bfcl` | Tool calling | AST match (`bfcl-eval`) | 100 | 25 | None |
| [ToolHop](toolhop.md) | `toolhop` | Tool calling | Answer accuracy | 100 | 100 | `TOOLHOP_ALLOW_DATASET_EXEC=1` |
| [API-Bank](api-bank.md) | `apibank` | Tool calling | API-call accuracy (API-Bank checkers) | 100 | 100 | None (source ships in the repo) |

The CLI name is the task's folder name and the optimizers' `--dataset` value. The launcher limit is the `--limit` that [`scripts/run_topologies.sh`](https://github.com/juyangbai/MAS-PromptBench/blob/main/scripts/run_topologies.sh) passes. The frozen IDs, 655 in total, are the instances behind the reported scores, listed in `benchmarks/<task>/<task>_eval_ids.json`.

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

Every task has one runner per topology variant, at `topologies/<topology>/[<framework>/]<task>/<framework>_<task>.py`. Run them from the repository root as modules, the way the sweep launcher does:

```bash title="Run a runner as a module"
python -m topologies.single.gpqa.langgraph_gpqa --batch --limit 100 \
  --out results/topologies_baseline/single_gpqa/predictions.jsonl
```

!!! warning "Run runners as modules"
    Most runners import the `topologies` package before they add the repository root to `sys.path`. Started as `python topologies/.../<file>.py` without `PYTHONPATH` set to the repository root, they stop with `ModuleNotFoundError: No module named 'topologies'`.

The runners come in three flag families:

| Tasks | Batch mode | Output | No-argument run |
| --- | --- | --- | --- |
| `gpqa`, `hotpotqa`, `math`, `lcb`, `apps` | needs `--batch` | `--out <file.jsonl>`; without it, scores are only printed | a built-in smoke demo |
| `bfcl`, `swe` | always; `--batch` is rejected | `--out-dir <dir>` | a small batch (5 BFCL `simple` instances, 2 SWE instances) |
| `toolhop`, `apibank` | always; `--batch` is accepted and ignored | `--out-dir <dir>` | a small batch (5 or 2 instances) |

All runners take `--limit`, `--offset` and `--only`. The launcher selects instances with `--limit` alone, which takes the first rows. To score exactly the frozen IDs, pass them to `--only`; each task page shows how.

!!! note "Launcher and BFCL or SWE"
    `scripts/run_topologies.sh` passes `--batch` to every runner and `--out` to all but ToolHop and API-Bank. The BFCL and SWE runners reject both flags, so run those two tasks directly as shown on their pages.

## Next steps

See [Evaluation Protocol](../evaluation/protocol.md) for how runs are scored and [Read the Results](../evaluation/results.md) for the files they write. For the multi-agent side of each run, start at [Workflow Topologies](../mas/topologies.md); for the optimizer, see [GEPA](../optimizers/gepa.md).
