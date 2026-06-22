# Communication Format

A study of a single question: **does the inter-agent communication *format* affect multi-agent performance?** It runs the existing `topologies/` runners under three communication formats and compares the scores — a control measured before any GEPA/MIPRO prompt optimization is applied.

## Overview

| Axis | Values |
|---|---|
| Topologies (4) | `independent`, `sequential`, `centralized`, `decentralized` |
| Datasets (5) | `hotpotqa`, `lcb`, `toolhop`, `apibank`, `swe` |
| Formats (3) | `freeform`, `semi_structured`, `structured_soft` |

### Directory layout

```
communications/
├── communication_formats.py   # engine: format contracts, prompt injection, parsing, run glue
├── output_contracts.py        # re-exports the scorer-facing contracts from topologies/
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

Each script is a thin wrapper that applies the chosen communication format on top of the matching topology runner, then exposes a standalone batch CLI:

```python
from communications.communication_formats import install_proxy, cli_main
install_proxy(globals(), topology="independent", dataset="hotpotqa", fmt="freeform")

if __name__ == "__main__":
    raise SystemExit(cli_main(globals()))
```

`install_proxy` loads the corresponding `topologies/<topology>/.../<framework>_<dataset>.py` runner, appends the chosen format's communication contract to the role prompts, wraps `solve` / `run_one` / `run_batch`, and exposes them in the script's namespace. `cli_main` makes the script runnable on its own (`python -m communications.<topology>.<dataset>.<dataset>_<format> --batch --limit N`). All real logic lives in the engine, **`communications/communication_formats.py`**.

Scoring is unchanged: only the *messages agents pass to each other* differ. Each run reuses the topology runner and the dataset's official scorer, and the final answer the scorer reads stays in the expected format — HotpotQA emits `Answer: <short-form>`, LCB the final fenced Python block, ToolHop `<answer>...</answer>`, and SWE-bench the repository diff.

---

## Formats

The format only governs the **inter-agent hand-off** — the report an agent passes to a peer, manager, next stage, or aggregator. The contract is appended to each agent's role prompt; the scorer-facing final artifact always comes *after* the report.

| Format | What agents are asked to emit |
|---|---|
| `freeform` | prompts unchanged (control) |
| `semi_structured` | a tagged report: required `[STATUS] [SUMMARY] [EVIDENCE_OR_TESTS] [CONFIDENCE] [NEXT]` + dataset-specific optional tags |
| `structured_soft` | a `JSON_REPORT: { status, summary, confidence, next, payload } END_JSON_REPORT` object (no code fences); falls back to raw text if it doesn't parse |

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

No malformed message is rejected or re-prompted — the experiment measures the *natural* parse-success rate and the resulting score delta, not enforced compliance.

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

Results land in `results/communications_baseline/<topology>_<dataset>_<format>/results.jsonl` (override with `--out`).

### Run the sweep

`scripts/run_communications.sh` runs every selected pair as its own process (no sharding):

```bash
TOPOLOGIES="independent centralized" \
DATASETS="hotpotqa toolhop" \
FORMATS="freeform structured_soft" \
bash scripts/run_communications.sh
```

Knobs: `VLLM_BASE_URL`, `MODEL_ID`, `TOPOLOGIES` / `DATASETS` / `FORMATS` (which pairs to sweep), and the per-dataset limits in the script.
