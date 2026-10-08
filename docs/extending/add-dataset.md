# Add a Dataset

A dataset is more than a loader: it needs a runner per topology, an output contract, frozen eval IDs, seed prompts, optimizer adapters and launcher entries. This page walks through each piece in the order you need it.
{ .lede }

The steps use `<ds>` for the new dataset's folder name. Use the same short name everywhere, because several components find the dataset by matching that string.

| Piece | Where | Read by |
| --- | --- | --- |
| Runners | `topologies/<topology>/[<framework>/]<ds>/<framework>_<ds>.py` | you, the sweeps, the optimizer adapters |
| Output contract | `topologies/output_contracts.py` (and three copies) | every runner and adapter |
| Eval IDs | `benchmarks/<ds>/<ds>_eval_ids.json` | the optimizers' train/val split |
| Seed prompts | `configs/prompts/<topology>/<ds>/<role>.txt` | runners, optimizers |
| Optimizer loader + adapters | `optimizers/{gepa,mipro}/real_runner_*/` | the GEPA and MIPRO pilots |
| Launcher entries | `scripts/*.sh`, `optimizers/*/run_*.sh` | sweeps |

## 1. Write one runner per topology

Each topology variant gets its own self-contained runner, eight files in all:

```text
topologies/single/<ds>/langgraph_<ds>.py
topologies/independent/<ds>/langgraph_<ds>.py
topologies/sequential/langgraph/<ds>/langgraph_<ds>.py
topologies/sequential/crewai/<ds>/crewai_<ds>.py
topologies/centralized/langgraph/<ds>/langgraph_<ds>.py
topologies/centralized/autogen/<ds>/autogen_<ds>.py
topologies/decentralized/langgraph/<ds>/langgraph_<ds>.py
topologies/decentralized/openai/<ds>/openai_<ds>.py
```

Start with the single-agent runner, following the layout of [`topologies/single/math/langgraph_math.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/single/math/langgraph_math.py):

1. **Config.** Module globals `VLLM_BASE_URL` and `MODEL_ID`, read from the environment. The optimizer adapters overwrite them by name.
2. **System prompt.** Read `configs/prompts/single/<ds>/<role>.txt` and wrap it with `append_output_contract_from_path(text, __file__, role)`. Multi-agent runners do this in `_load_prompt(role)`.
3. **Agent.** `build_agent()` (`_build_llm()` or `_build_client()` in multi-agent runners) builds the client from those globals.
4. **Scoring.** Copy the dataset's official scorer verbatim, as the MATH runners do with Hendrycks' `is_equiv`.
5. **`solve(...)`.** Runs one instance; returns a dict with at least `answer` and `raw`.
6. **`load_instances(limit=None, offset=0, only=None)`.** Returns dicts with a stable string `id`. Apply `only` before `offset` and `limit`.
7. **`run_batch(instances, out_path=None)`.** Writes one JSON line per row (ID, gold, prediction, score, latency, telemetry from `topologies/telemetry.py`, error) and returns a summary.
8. **`_canned_demo()` and the CLI.** A no-argument run calls the demo; `--batch` runs the dataset.

Pick a CLI shape from an existing family (see [Command-Line Flags](../reference/cli.md)). The `--batch` / `--out` family is the simplest and is what `scripts/run_topologies.sh` expects by default:

```python title="CLI block, condensed from topologies/single/math/langgraph_math.py"
parser.add_argument("--batch", action="store_true")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--only", nargs="*", default=None)
args = parser.parse_args()
if not args.batch:
    _canned_demo()
    sys.exit(0)
instances = load_instances(limit=args.limit, offset=args.offset, only=args.only)
run_batch(instances, out_path=Path(args.out) if args.out else None)
```

Then port the agent wiring to the other seven runners, keeping the loader and scorer unchanged so scores stay comparable.

## 2. Register the output contract

Add one sentence to `DATASET_CONTRACTS` in [`topologies/output_contracts.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/topologies/output_contracts.py) that names the final artifact your scorer reads, for example "End with exactly one final line: Answer: <short-form>." Then list the roles that emit that artifact:

- single and independent: roles in `PRIMARY_FINAL_ROLES` (`solver`, `caller`, `coder`, `patcher`, `planner`).
- decentralized: `debater`.
- sequential: add `"<ds>"` to `SEQUENTIAL_FINAL_ROLES`.
- centralized: add `"<ds>"` to `CENTRALIZED_FINAL_ROLES_BY_DATASET`.

`append_output_contract_from_path` finds the dataset by matching a folder in the runner's path against the `DATASET_CONTRACTS` keys, so the key must equal `<ds>`. Repeat the change in `teamsizes/output_contracts.py` and in the copies under `optimizers/gepa/real_runner_gepa/` and `optimizers/mipro/real_runner_mipro/`, whose entries also carry the `PROTECTED FINAL OUTPUT CONTRACT:` header. `communications/output_contracts.py` re-exports the topologies version.

## 3. Freeze the eval IDs

Write `benchmarks/<ds>/<ds>_eval_ids.json` in the same shape as the existing manifests:

```json title="benchmarks/<ds>/<ds>_eval_ids.json"
{
  "dataset": "<ds>",
  "sample": "report 100",
  "n": 100,
  "source": "how the IDs were chosen",
  "ids": ["first_id", "second_id"]
}
```

The `ids` must equal the `id` values from `load_instances`. Runners other than API-Bank do not read this file, so pass the IDs with `--only` to score the reported set. The optimizers read it to keep these IDs out of their train and validation splits.

## 4. Generate the seed prompts

Seed prompts are written by an LLM. For every `(topology, dataset, role)` in `roles.yaml`, [`configs/generate_role_prompts.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/configs/generate_role_prompts.py) fills `meta_prompt.txt` with the dataset line from `domains.yaml`, the topology and role descriptions from `roles.yaml` and the tool list from `tools.yaml`, sends it as one user message, strips any `<think>` block and writes the reply to `configs/prompts/<topology>/<ds>/<role>.txt`.

Add the dataset to the three YAML files in `configs/prompts/`:

- `domains.yaml`: one line, `<ds>: "..."`. The generator only visits datasets listed here.
- `roles.yaml`: under `topologies.<topology>.benchmarks` for each of the five topologies, add `<ds>:` with `<role>: "<job description>"`. Role names must match what your runners load.
- `tools.yaml`: under each topology, `<ds>:` with the tool list (anchors such as `*calculator` work). A missing entry becomes `None.`, which tells the prompt to use no tools.

Then generate. Existing prompt files are skipped unless you pass `--force`:

```bash title="Write the new prompt files"
export PROMPT_GEN_BASE_URL=http://localhost:8000/v1   # serves the generator model
python configs/generate_role_prompts.py --only single/<ds> --only independent/<ds> \
  --only sequential/<ds> --only centralized/<ds> --only decentralized/<ds>
```

The generator defaults to `Qwen/Qwen3.5-122B-A10B-FP8` with temperature 0 and seed 42. The `decentralized_openai` block in `roles.yaml` only covers MATH; the OpenAI-SDK runners for other datasets load `configs/prompts/decentralized/<ds>/`.

## 5. Wire up the optimizers

Each optimizer needs a dataset module, adapters and a registry entry. In `optimizers/gepa/real_runner_gepa/`:

- **`datasets/<ds>.py`**, from [`templates/dataset_template.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/optimizers/gepa/templates/dataset_template.py). `load_all()` returns `dspy.Example` objects with an `id` and a `task_instance` input; `train_val_split(...)` should call `split_utils.train_val_split_excluding_real_eval("<ds>", ...)` so eval IDs stay out. Add the `metric(example, prediction, ...)` the template lacks, returning `dspy.Prediction(score=..., feedback=...)` as `datasets/math.py` does. The pilot imports `real_runner_gepa.datasets.<dataset>`, so the file name must equal the registry key.
- **`adapters/module_<ds>.py`**, one class per optimizer topology name. Subclass `ModuleAdapterBase` (`adapters/module_common.py`) as `module_lcb.py` does, set `topology`, `framework`, `prompt_topology`, `roles_` and `module_name`, and implement `run_example` to call your runner's `solve`.
- **`registry.py`**: add `"<ds>": {...}` to `DATASET_ADAPTERS` with all eight topology names mapped to `"real_runner_gepa.adapters.module_<ds>:<Class>"`.

Repeat under `optimizers/mipro/real_runner_mipro/`, which has no `templates/` folder. [Add a Topology](add-topology.md) covers the adapter contract and smoke test in detail. A minimal run checks the wiring:

```bash title="Smoke-test the GEPA wiring"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset <ds> --topology single \
  --train-size 1 --val-size 1 --max-full-evals 1 --out results/gepa/single_<ds>_smoke
```

## 6. Add it to the launchers

The sweep scripts hard-code their dataset list and per-dataset row limits:

```bash title="scripts/run_topologies.sh (LIMIT wrapped)"
DATASETS="${DATASETS:-gpqa hotpotqa math lcb apps bfcl swe apibank toolhop}"
declare -A LIMIT=([gpqa]=100 [hotpotqa]=100 [math]=100 [lcb]=50 [apps]=50
                  [bfcl]=100 [swe]=30 [apibank]=100 [toolhop]=100)
```

Add `<ds>` to both; a dataset missing from `LIMIT` runs 50 rows. The script passes `--batch --limit N --out <file>` to every dataset except `apibank` and `toolhop`, so a runner that takes `--out-dir` needs its own `case` branch. Make the same edits in `scripts/run_teamsizes.sh` and `scripts/run_communications.sh` if you add those variants, and add `<ds>` to `DATASETS` in `optimizers/gepa/run_gepa.sh` and `optimizers/mipro/run_mipro.sh`.

## Checklist

1. Eight runners sharing one loader and scorer.
2. `DATASET_CONTRACTS` and final roles in all four `output_contracts.py` copies.
3. `benchmarks/<ds>/<ds>_eval_ids.json` matching the runner IDs.
4. YAML entries and generated prompts under `configs/prompts/<topology>/<ds>/`.
5. `datasets/<ds>.py` (`load_all`, `train_val_split`, `metric`) for GEPA and MIPRO.
6. `adapters/module_<ds>.py` and `DATASET_ADAPTERS["<ds>"]` in both registries.
7. `DATASETS` and `LIMIT` entries in the launchers.
8. A smoke demo, a short `--batch --limit` slice per runner and a one-row optimizer run.
