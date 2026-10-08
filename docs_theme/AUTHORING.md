# Writing MAS-PromptBench docs pages

These docs are a tutorial and reference for the MAS-PromptBench repository
(github.com/juyangbai/MAS-PromptBench). Pages are Markdown, built with MkDocs and a
custom theme. Read `docs/getting-started/installation.md` as the model page before writing.

## Voice

- Plain, direct, accurate. Write for a researcher who wants to run things.
- Short sentences, active voice. Second person ("Run…", "Set…").
- No marketing words, no emoji, no em-dash asides, no "Note that", no "worth noting".
- Never invent: every command, flag, path, env var, default, number and file name must be
  checked against the code. If you can't confirm something, leave it out.
- Don't cite paper results; these docs cover how to run the benchmark, not its findings.

## Page shape

```markdown
# Page Title

One or two sentences saying what this page covers and why it matters.
{ .lede }

## First section
...
```

- Exactly one `# H1`, then the `{ .lede }` paragraph. Use `##` and `###` below it.
- Aim for 300–900 words. Lead with what the reader needs to do; put detail after.
- Link to sibling pages with relative links to the `.md` file, e.g.
  `[Sequential](../mas/sequential.md)` or `[GEPA](gepa.md#settings)`. Only link to
  pages that exist in `mkdocs.yml`.
- Link to repo files with full GitHub URLs:
  `https://github.com/juyangbai/MAS-PromptBench/blob/main/<path>` (files) or
  `.../tree/main/<path>` (folders).

## Markdown features available

**Code** — fenced, with a language and usually a title:

````markdown
```bash title="Run a baseline"
python topologies/single/hotpotqa/langgraph_hotpotqa.py --batch --limit 100 \
  --out results/topologies_baseline/single_hotpotqa/predictions.jsonl
```
````

Use `bash`, `python`, `json`, `yaml`, `text`. Keep lines under ~90 characters (wrap
with `\`). Comments are welcome but short.

**Callouts** (admonitions). Use at most two or three per page:

```markdown
!!! tip "Short bold lead-in"
    Body text, one to three sentences.

!!! warning "Heads-up"
    Something that will break or mislead if ignored.

!!! note "Background"
    Context that is useful but optional.
```

**Tabs** — for alternatives (GEPA vs MIPRO, LangGraph vs CrewAI, API vs local):

```markdown
=== "LangGraph"

    ```bash
    python topologies/sequential/langgraph/bfcl/langgraph_bfcl.py --limit 100
    ```

=== "CrewAI"

    ```bash
    python topologies/sequential/crewai/bfcl/crewai_bfcl.py --limit 100
    ```
```

**Tables** — standard pipe tables. Put code in backticks.

**Fact strip** — a row of key facts at the top of a detail page (task, topology, method).
Use 3–5 items, short values:

```markdown
<div class="facts" markdown>
<div><span>Domain</span>Reasoning</div>
<div><span>Metric</span>Exact match</div>
<div><span>Default limit</span>100</div>
<div><span>Eval IDs</span>100</div>
</div>
```

**Card grid** — links to child pages (overview pages only):

```markdown
<div class="cards" markdown>

- [GPQA-Diamond](gpqa.md)
  Graduate-level science multiple choice.
- [HotpotQA](hotpotqa.md)
  Multi-hop question answering over Wikipedia.

</div>
```

Each list item becomes a clickable card: the link is the title, the next line the
description (one sentence). Keep the blank lines around the list.

**Diagrams** — topology diagrams already exist as snippets; include one with
`--8<-- "diagrams/sequential.svg"` on its own line (names: single, independent,
sequential, centralized, decentralized). Don't draw other diagrams.

**Math** — `\( \Delta \)` inline, `\[ ... \]` display.

## Verified facts (start here, then confirm details in code)

### Runner command shapes

- **Run every runner as a module from the repository root**:
  `python -m topologies.single.hotpotqa.langgraph_hotpotqa`. Most runner files import the
  `topologies` package before adding the repo root to `sys.path`, so the file-path form
  (`python topologies/...py`) fails with `ModuleNotFoundError` unless `PYTHONPATH=.` is set.
- Topology runners: `topologies/<topology>/[<framework>/]<dataset>/<framework>_<dataset>.py`.
  `single` and `independent` are LangGraph-only (no framework folder). `sequential`:
  langgraph, crewai. `centralized`: langgraph, autogen. `decentralized`: langgraph,
  openai (OpenAI SDK). Every topology × dataset pair exists (72 scripts).
- Datasets (folder/CLI names): `gpqa`, `hotpotqa`, `math`, `lcb` (LiveCodeBench),
  `apps`, `swe` (SWE-bench Verified), `bfcl`, `toolhop`, `apibank`.
- Flags differ by runner family:
  - gpqa / hotpotqa / math / lcb / apps: `--batch`, `--limit`, `--offset`, `--only`,
    `--out <file.jsonl>`. Without `--out`, a batch only prints scores. No-arg run = smoke demo.
  - bfcl, swe: no `--batch` flag (they always run a batch; passing `--batch` errors).
    Take `--limit` (bfcl default 5, swe default 2), `--offset`, `--only`, `--out-dir <dir>`.
    bfcl has `--category` (default `simple`). swe has `--eval` (`local|singularity|none`
    in the single runner, default `local`; `singularity|none` elsewhere, default
    `singularity`), `--workdir-root`, `--keep-workdirs`.
  - toolhop, apibank: accept `--batch` (ignored, always batch), `--limit` (toolhop 5,
    apibank 2), `--offset`,
    `--only`, `--out-dir`. apibank has `--level`, `--summary`, `--curated-path`.
    toolhop needs `export TOOLHOP_ALLOW_DATASET_EXEC=1`.
- Communication-protocol runners: `python -m communications.<topology>.<dataset>.<dataset>_<format>`
  for topologies independent/sequential/centralized/decentralized, datasets
  hotpotqa/lcb/swe/toolhop/apibank, formats `freeform`, `semi_structured`,
  `structured_soft` (shown to readers as Freeform / Semi-structured / Structured).
  Shared CLI (`communications/communication_formats.py: cli_main`): `--batch`, `--limit`,
  `--offset`, `--only`, `--out`. Default output:
  `results/communications_baseline/<topology>_<dataset>_<format>/results.jsonl`.
- Team-size runners: `teamsizes/<topology>/<dataset>/<dataset>_r<N>.py`, N ∈ {2,4,8,10},
  topologies independent/sequential/centralized/decentralized, all 9 datasets. Same flag
  families as topology runners (bfcl/swe: no `--batch`, `--out-dir`; toolhop/apibank:
  `--out-dir`; others `--batch` + `--out`). Their default output paths don't include the
  team size, so pass an explicit `--out`/`--out-dir` such as
  `results/teamsizes_r4/hotpotqa/centralized_r4/`.
- Sweep launchers: `scripts/run_topologies.sh`, `scripts/run_communications.sh`,
  `scripts/run_teamsizes.sh` (env: `VLLM_BASE_URL`, `MODEL_ID`, `DATASETS`, `TOPOLOGIES`,
  `FORMATS`, `RVALUES`, `OUT_ROOT`). Known issue: `run_topologies.sh` and
  `run_teamsizes.sh` pass `--batch` to bfcl/swe runners, which rejects it.
- Launcher limits: gpqa 100, hotpotqa 100, math 100, lcb 50, apps 50, bfcl 100, swe 30,
  apibank 100, toolhop 100.

### Optimizers

- GEPA: `cd optimizers/gepa && python -m real_runner_gepa.pilots.run_gepa_dataset --dataset <ds> --topology <name> --train-size 25 --val-size 25 --max-full-evals 5 --out results/gepa/<name>_<ds>`.
- MIPRO: `cd optimizers/mipro && python -m real_runner_mipro.pilots.run_mipro_dataset --dataset <ds> --topology <name> --train-size 25 --val-size 25 --num-candidates 3 --num-trials 3 --out results/mipro/<name>_<ds>`.
- Optimizer topology names: `single`, `independent`, `sequential`, `sequential_crewai`,
  `centralized`, `centralized_autogen`, `decentralized`, `decentralized_openai`; plus
  `<topo>_r<N>` and `<topo>_communications_<format>` **only for** hotpotqa, lcb, toolhop,
  apibank. Check with `real_runner_gepa.registry.topologies(dataset)`.
- Env: `GEPA_TASK_ENDPOINTS` / `MIPRO_TASK_ENDPOINTS` (comma-separated, runs the agents),
  `GEPA_REFL_ENDPOINT` / `MIPRO_REFL_ENDPOINT` (reflection / proposal model), `TASK_MODEL`,
  `REFL_MODEL` (MIPRO also reads `MIPRO_TASK_MODEL`, `MIPRO_REFL_MODEL`),
  `GEPA_EXCLUDE_REAL_EVAL_IDS` / `MIPRO_EXCLUDE_REAL_EVAL_IDS` (**on by default**: unset
  counts as on; `0/false/no/off` turns it off),
  `OPENAI_API_KEY`. The optimizers do not read `VLLM_BASE_URL` for task calls; unset
  endpoint vars fall back to built-in localhost defaults in `lm.py`.
- The pilots default to `--n-agents 2 --n-rounds 1` (used by independent and decentralized
  cells). Pass `--n-agents 4 --n-rounds 2` to optimize the same team the runners use.
- GEPA needs DSPy 3 (`dspy.teleprompt.GEPA`); `environment.yml` pins `dspy<3`, so readers
  must run `pip install -U "dspy[optuna]>=3"`.
- Defaults in code: TASK_MODEL `Qwen/Qwen3.5-9B`, REFL_MODEL `Qwen/Qwen3.5-122B-A10B-FP8`.
- Output dir contains `meta.json` (`baseline_score`, `compiled_score`, `delta`,
  `accept_compiled_prompt`, `selected_prompt_source`, `compiled_prompt_files`, …),
  `baseline_val.jsonl`, `compiled_val.jsonl`, `optimized_val.jsonl`, `compiled/<name>.txt`,
  `compiled_raw/`, `status.json`. Scores are fractions in [0, 1]. The compiled prompt is
  accepted when `compiled_score + eps >= baseline_score`, otherwise the baseline prompt is kept.
- Seed prompts: `configs/prompts/<topology>/<dataset>/<role>.txt` (read-only for optimizers).

### Model connection

- All runners use one OpenAI-compatible endpoint: `VLLM_BASE_URL`, `MODEL_ID`,
  `OPENAI_API_KEY` (runners default the key to `EMPTY`).
- `models/serve_qwen3_5_9b.sh` (one replica per GPU from port 8000; `VLLM_BASE_PORT`),
  `models/serve_qwen3_5_122b.sh` (TP=4, FP8-capable GPUs).

## Don'ts

- Keep each page to its own topic; check facts against the code before you publish.
- Don't add pages to `mkdocs.yml`; don't create extra files (except in your own folder if asked).
- No placeholder text, no TODOs, no lorem ipsum.
