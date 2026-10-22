# Writing MAS-PromptBench docs pages

These docs are a tutorial and reference for the MAS-PromptBench repository
(github.com/juyangbai/MAS-PromptBench). Pages are Markdown, built with MkDocs and a
custom theme. Read `docs/content/getting-started/installation.md` as the model page before writing.

## Voice

- Plain, direct, accurate. Write for a researcher who wants to run things.
- Short sentences, active voice. Second person ("Run…", "Set…").
- No marketing words, no emoji, no em-dash asides, no "Note that", no "worth noting".
- Never invent: every command, flag, path, env var, default, number and file name must be
  checked against the code. If you can't confirm something, leave it out.
- Don't cite benchmark results or paper numbers; the docs describe how to run and read the
  benchmark, not its findings.

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
  `[Sequential](../mas/sequential.md)` or `[Flags](swe-bench.md#flags)`. Only link to
  pages that exist in `docs/mkdocs.yml`.
- Link to repo files on GitHub:
  `https://github.com/juyangbai/MAS-PromptBench/blob/main/<path>` (files) or `https://github.com/juyangbai/MAS-PromptBench/tree/main/<path>` (folders).

## Markdown features available

**Code** — fenced, with a language and usually a title:

````markdown
```bash title="Run a baseline"
python -m topologies.single.hotpotqa.langgraph_hotpotqa --batch --limit 100 \
  --out-dir results/topologies_baseline/single_hotpotqa
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

!!! takeaway
    The one-sentence lesson of a section.
```

**Tabs** — for alternatives (LangGraph vs CrewAI, one cell vs another, local vs remote):

```markdown
=== "LangGraph"

    ```bash
    python -m topologies.sequential.langgraph.bfcl.langgraph_bfcl --limit 100
    ```

=== "CrewAI"

    ```bash
    python -m topologies.sequential.crewai.bfcl.crewai_bfcl --limit 100
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
  `python -m topologies.single.hotpotqa.langgraph_hotpotqa`. The runners import the shared
  `core` package, so the file-path form fails with `ModuleNotFoundError: No module named
  'core'` unless `PYTHONPATH=.` is set.
- Topology runners: `topologies/<topology>/[<framework>/]<dataset>/<framework>_<dataset>.py`.
  `single` and `independent` are LangGraph-only (no framework folder). `sequential`:
  langgraph, crewai. `centralized`: langgraph, autogen. `decentralized`: langgraph,
  openai_agents (OpenAI Agents SDK). Every topology × dataset pair exists (72 modules).
  Shared code (CLI, batch loop, clients, settings, team specs, prompts, one task module per
  dataset in `core/tasks/`) lives in `core/`.
- Datasets (folder/CLI names): `gpqa`, `hotpotqa`, `math`, `lcb` (LiveCodeBench),
  `apps`, `swe` (SWE-bench Verified), `bfcl`, `toolhop`, `apibank`.
- One command line for every runner (`core/cli.py`): `--batch`, `--limit N`, `--offset K`,
  `--only ID ...` (overrides `--limit`), `--out-dir DIR` (writes `DIR/predictions.jsonl`),
  `--out PATH`. Output files are emptied when a batch starts. Dataset options: gpqa
  `--shuffle-seed`; lcb `--difficulty` (`--platform` on LangGraph sequential, centralized,
  decentralized); apps `--difficulty`, `--max-tests-per-row` (default 20); bfcl `--category`
  (default `simple`); swe `--eval` (`local|singularity|none` in the single runner, default
  `local`; `singularity|none` elsewhere, default `singularity`), `--workdir-root`,
  `--keep-workdirs`, `--subset`; toolhop `--smoke-dataset`; apibank `--level`, `--summary`,
  `--curated-path`, `--toolsearcher-scorer`.
- Without `--batch`, gpqa / hotpotqa / math / lcb / apps runners play a canned demo; bfcl,
  swe, toolhop and apibank have no demo and run a batch (default `--limit` 5, 2, 5, 2).
- Eval IDs: `benchmarks/<ds>/<ds>_eval_ids.json` (730 in all); fixed splits:
  `benchmarks/<ds>/<ds>_splits.json` (`test` = eval IDs). `--limit N` selects exactly the
  eval IDs for every dataset except bfcl (stratified 100: 40/20/20/20) and swe (fixed 30),
  which take `--only`.
- Team specs: `configs/teams/<dataset>.yaml` (all but toolhop and apibank), team size
  r ∈ {2, 4, 8, 10}, r = 4 for the `topologies/` runners.
- Communication-protocol runners: `python -m communications.<topology>.<dataset>.<dataset>_<format>`
  for topologies independent/sequential/centralized/decentralized, datasets
  hotpotqa/lcb/bfcl/toolhop/apibank/swe, formats `freeform`, `semi_structured`,
  `structured_soft` (shown to readers as Freeform / Semi-structured / Structured); 72 runners.
  Default output: `results/communications_baseline/<topology>_<dataset>_<format>/results.jsonl`.
- Team-size runners: `teamsizes/<topology>/<dataset>/<dataset>_r<N>.py`, N ∈ {2,4,8,10},
  topologies independent/sequential/centralized/decentralized, all 9 datasets (144). They
  run the LangGraph runner with a preset team (`core/variant.py`); toolhop and apibank vote
  over N replicas (`teamsizes/<ds>_common.py`).
- Sweep launchers: `scripts/run_topologies.sh`, `scripts/run_communications.sh`,
  `scripts/run_teamsizes.sh` (env: `VLLM_BASE_URL`, `MODEL_ID`, `DATASETS`, `TOPOLOGIES`,
  `FORMATS`, `RVALUES`, `OUT_ROOT`). Don't run them to check a page; they start real sweeps.
- The OpenAI Agents SDK is installed apart:
  `pip install --target vendor/openai_agents -r requirements-openai-agents.txt`. Its runners
  re-exec with it first on `PYTHONPATH`.

### Optimizers

- Eight methods, presented equally: GEPA, MIPRO, MAPRO, MASPO, HiveMind, MAMUT-GEPA,
  MASPOB, TAVO (keys `gepa`, `mipro`, `mapro`, `maspo`, `hivemind`, `mamut_gepa`, `maspob`,
  `tavo`), one package each under `optimizers/`.
- One run protocol: `python -m optimizers.protocol.run --method <key> --dataset <ds>
  --topology <topology> [--framework F] [--team-size N] [--communication FMT] --model qwen
  --seed {0,1,2} --out runs/...`; `python -m optimizers.protocol.aggregate runs/`.
- Env: `TASK_ENDPOINTS` (else `VLLM_BASE_URL`), `REFLECTION_MODEL_BASE_URL` (default
  `http://localhost:8200/v1`). Reflection model `Qwen/Qwen3.5-122B-A10B-FP8`.
- Cells outside the experiment grid (`optimizers/protocol/cells.py`) need `--allow-any-cell`.
- A job writes `result.json` under `--out`; `test.delta_pp` is Δ in percentage points.
- Seed prompts: `configs/prompts/<topology>/<dataset>/<role>.txt` (read-only for optimizers).

### Model connection

- All runners use one OpenAI-compatible endpoint: `VLLM_BASE_URL` (default
  `http://localhost:8000/v1`), `MODEL_ID` (default `Qwen/Qwen3.5-9B`), `OPENAI_API_KEY`
  (default `EMPTY`). Decoding: temperature 0.0, top-p 0.9, at most 32,768 output tokens.
- `models/serve_qwen3_5_9b.sh` (one replica per GPU from port 8000),
  `models/serve_llama3_1_8b.sh` (gated; from port 8100), `models/serve_qwen3_5_122b.sh`
  (reflection model; TP=4, port 8200).

## Don'ts

- Keep each page to its own topic; check facts against the code before you publish.
- Don't add pages to `docs/mkdocs.yml`; don't create extra files (except in your own folder if asked).
- No placeholder text, no TODOs, no lorem ipsum.
