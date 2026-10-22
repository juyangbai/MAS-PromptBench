# Communication Format

A study of a single question: **does the inter-agent communication *format* affect multi-agent performance?** It runs the `topologies/` LangGraph runners under three communication formats and compares the scores — a control measured before any prompt optimization is applied.

## Overview

| Axis | Values |
|---|---|
| Topologies (4) | `independent`, `sequential`, `centralized`, `decentralized` |
| Datasets (6) | `hotpotqa`, `lcb`, `bfcl`, `toolhop`, `apibank`, `swe` |
| Formats (3) | `freeform`, `semi_structured`, `structured_soft` |

### Directory layout

```
communications/
├── communication_formats.py   # the pairs: runner loading, report scoring, records, CLI
├── output_contracts.py        # re-exports the scorer-facing contracts from core/
│
├── independent/               # one directory per topology
│   └── <dataset>/<dataset>_<format>.py
├── sequential/
├── centralized/
└── decentralized/
```

Path pattern: `communications/<topology>/<dataset>/<dataset>_<format>.py`.

---

## How it works

Each script is a thin wrapper that declares one pair — its topology, dataset and format:

```python
from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="hotpotqa", fmt="freeform")
```

`install` loads the matching LangGraph runner (`topologies/<topology>/[langgraph/]<dataset>/langgraph_<dataset>.py`) as a module of its own with `COMMUNICATION_FORMAT` preset, so pairs never interfere. The runner's `COMMUNICATION` policy (**`core/communication.py`**) appends the format's contract to the agents' system prompts; the HotpotQA and LCB runners, and the sequential and centralized BFCL runners, also render each hand-off in the format. `install` adds `solve` (hand-offs recorded, reports scored), `run_one` / `run_batch` and the `core.cli` command line.

Scoring is unchanged: only the *messages agents pass to each other* differ. Each run reuses the topology runner and the dataset's official scorer, and the final answer the scorer reads stays in the expected format — HotpotQA emits `Answer: <short-form>`, LCB the final fenced Python block, BFCL the final fenced JSON call list, ToolHop `<answer>...</answer>`, API-Bank one bracketed API call (`[ApiName(arg='value')]`), and SWE-bench the repository diff.

---

## Formats

The format only governs the **inter-agent hand-off** — the report an agent passes to a peer, manager, next stage, or aggregator; the scorer-facing final artifact always comes *after* the report.

| Format | What agents are asked to emit |
|---|---|
| `freeform` | prompts unchanged (control) |
| `semi_structured` | a tagged report: required `[STATUS] [SUMMARY] [EVIDENCE_OR_TESTS] [CONFIDENCE] [NEXT]` + dataset-specific optional tags |
| `structured_soft` | a `JSON_REPORT: { status, summary, confidence, next, payload } END_JSON_REPORT` object (no code fences) |

### Example of Communication Formats

The three formats produce the inter-agent reports below for the HotpotQA question *"Which magazine started first — Arthur's Magazine or First for Women?"* (gold answer: *Arthur's Magazine*). In every case the scorer-facing `Answer:` line follows the report unchanged.

**`freeform`**

```text
Arthur's Magazine was founded in 1844; First for Women in 1989, so Arthur's
Magazine came first.
Answer: Arthur's Magazine
```

**`semi_structured`**

```text
[STATUS]
completed
[SUMMARY]
Arthur's Magazine (1844) predates First for Women (1989).
[EVIDENCE_OR_TESTS]
Arthur's Magazine — founded 1844; First for Women — founded 1989.
[CONFIDENCE]
high — both founding years confirmed from Wikipedia.
[NEXT]
Use Arthur's Magazine as the final answer.
[ANSWER_CANDIDATE]
Arthur's Magazine

Answer: Arthur's Magazine
```

**`structured_soft`**

```text
JSON_REPORT:
{
  "status": "completed",
  "summary": "Arthur's Magazine (1844) predates First for Women (1989).",
  "confidence": "high",
  "next": "Use Arthur's Magazine as the final answer.",
  "payload": {"entities": ["Arthur's Magazine", "First for Women"],
              "answer_candidate": "Arthur's Magazine"}
}
END_JSON_REPORT

Answer: Arthur's Magazine
```

No malformed message is rejected or re-prompted: each record's parse metrics (`communication_parse_rate`, ...) judge the agents' own text, so the experiment measures the *natural* parse-success rate and the resulting score delta, not enforced compliance.

---

## Usage

### Point at an endpoint

```bash
export VLLM_BASE_URL=http://localhost:8000/v1   # any OpenAI-compatible server
export MODEL_ID=Qwen/Qwen3.5-9B
export TOOLHOP_ALLOW_DATASET_EXEC=1             # required for ToolHop pairs
```

### Run a baseline

Each pair is a standalone batch runner:

```bash
python -m communications.independent.hotpotqa.hotpotqa_freeform --batch --limit 100
```

Results land in `results/communications_baseline/<topology>_<dataset>_<format>/results.jsonl` (BFCL: `.../<category>/results.jsonl`, per `--category`); override with `--out` or `--out-dir` ([options](../docs/content/reference/cli.md#runner-options)).

### Run the sweep

`scripts/run_communications.sh` runs every selected pair as its own process (no sharding):

```bash
TOPOLOGIES="independent centralized" \
DATASETS="hotpotqa toolhop" \
FORMATS="freeform structured_soft" \
bash scripts/run_communications.sh
```

Knobs: `TOPOLOGIES` / `DATASETS` / `FORMATS` (which pairs to sweep), plus the endpoint variables above. BFCL and SWE-bench pairs run their [evaluation IDs](../benchmarks/README.md); the others use the per-dataset `--limit` in the script ([sweep launchers](../docs/content/reference/environment.md#sweep-launchers)).
