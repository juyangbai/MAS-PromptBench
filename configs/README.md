# Initial System Prompts

This folder holds every agent's **initial system prompt** — one per `(topology, dataset, role)`. The prompts are **written by an LLM from a structured definition** (the YAML files below), not manually, and serve as the seeds that the prompt optimizers later optimize. At runtime, each topology runner loads its prompts from `configs/prompts/<topology>/<dataset>/<role>.txt` (`core/prompts.py` adds the role's protected final-output contract).

## Layout

Four source files define the spec; the generator writes one prompt file per `(topology, dataset, role)` (**225 total** over 5 topologies × 9 datasets).

```
configs/
├── generate_role_prompts.py            # the generator (run to reproduce)
├── teams/<dataset>.yaml                # team specs (see Team specs below)
└── prompts/
    ├── meta_prompt.txt                 # template: {DOMAIN_BACKGROUND} {TOPOLOGY_DESC} {TOOLS} {ROLE} {ROLE_DESC}
    ├── domains.yaml                    # one-line domain description per dataset
    ├── roles.yaml                      # catalog: topology → benchmarks → <dataset> → <role>: <description>
    ├── tools.yaml                      # tool catalog per (topology, dataset)
    └── <topology>/<dataset>/<role>.txt # generated role system prompts
```

| Source file | Defines |
|---|---|
| `meta_prompt.txt` | the prompt-writing template, with placeholders filled in per pair |
| `domains.yaml` | the one-line domain description for each dataset |
| `roles.yaml` | every `(topology, dataset, role)` tuple and its one-line job description |
| `tools.yaml` | the tools each pair's agents may reference (YAML anchors for reuse) |

---

## How prompts are generated

For each `(topology, dataset, role)` in `roles.yaml`, `configs/generate_role_prompts.py`:

1. **Renders** `meta_prompt.txt` with that pair's domain (`domains.yaml`), topology + role description (`roles.yaml`), and tool list (`tools.yaml`).
2. **Asks an LLM** to write the system prompt — the rendered template is sent as one user message, and the reply *is* the prompt.
3. **Cleans and writes** — strips Qwen `<think>…</think>` reasoning, then writes `configs/prompts/<topology>/<dataset>/<role>.txt`.

Design notes:

- **Authored by a stronger model than runtime.** Generation uses Qwen3.5-122B while dataset runs use the 9B; exact settings are in [Configuration](#configuration).
- **Tool-faithful.** The meta-prompt forbids referencing any tool not listed for the pair (no invented web search / code executors); if the tool list is `None`, the agent must reason directly from the prompt.
- **Seeds, not finals.** `tools.yaml` entries are intentionally minimal to leave room for the downstream optimizer.

---

## Reproducing the prompts

### Setup

- The `mas-promptbench` conda env (provides `openai` + `pyyaml`).
- An OpenAI-compatible endpoint serving the generator model (default Qwen3.5-122B):

```bash
bash models/serve_qwen3_5_122b.sh
export PROMPT_GEN_BASE_URL=http://localhost:8200/v1
```

### Configuration

The committed prompts were authored with the defaults below. Each is overridable, but reproducing the existing prompts exactly requires keeping them:

| Setting | Default | Override |
|---|---|---|
| Model | `Qwen/Qwen3.5-122B-A10B-FP8` | `PROMPT_GEN_MODEL` or `--model` |
| Endpoint | `http://localhost:8200/v1` | `PROMPT_GEN_BASE_URL` or `--base-url` |
| API key | `EMPTY` | `PROMPT_GEN_API_KEY` or `--api-key` |
| Temperature | `0.0` | `--temperature` |
| Seed | `42` | `--seed` |

Generation is deterministic (`temperature = 0`, fixed `seed`), so re-running with these defaults reproduces byte-identical prompts. The generator reads only the `PROMPT_GEN_*` variables and ignores the runtime `VLLM_BASE_URL` / `MODEL_ID`, so a dataset run cannot flip prompt authoring onto the smaller model.

### Generate

```bash
# write any missing prompts (skips existing files)
python configs/generate_role_prompts.py

# overwrite all prompts
python configs/generate_role_prompts.py --force

# restrict to a prefix: <topology>[/<dataset>[/<role>]]
python configs/generate_role_prompts.py --only sequential/gpqa
python configs/generate_role_prompts.py --only single/gpqa --only independent/gpqa
```

---

## Team specs

[`teams/`](teams/README.md) declares each dataset's team for every multi-agent topology and team size `r ∈ {2, 4, 8, 10}`: the `topologies/` runners (all but CrewAI) use `r = 4`, the [`teamsizes/`](../teamsizes/README.md) runners every r.
