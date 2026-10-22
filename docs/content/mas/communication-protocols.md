# Communication Protocols

The same topology can pass messages as free text, as tagged reports or as JSON. This page covers the three message formats, which cells use them, and how to run and optimize them.
{ .lede }

<div class="facts" markdown>
<div><span>Formats</span>3</div>
<div><span>Topologies</span>4</div>
<div><span>Datasets</span>6</div>
<div><span>Runners</span>72</div>
</div>

## What a format changes

A format governs only the inter-agent hand-off: the report an agent passes to the next stage, the manager, a peer or the aggregator. The model, topology, roles, tools and scorer stay the same, and the scorer-facing final answer keeps its usual form after the report.

Each topology runner applies the format itself through its communication policy, defined in [`core/communication.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/core/communication.py), in two places:

1. **In the prompts.** A format contract is appended to every agent's system prompt, telling it how to write its report and to put the final artifact after it.
2. **In the hand-offs.** Before a receiver reads another agent's output, the runner re-renders that text in the chosen format, so the receiver always gets a well-formed message. The HotpotQA and LiveCodeBench runners, the Sequential and Centralized BFCL runners, and the ToolHop and API-Bank runners do this; the SWE-bench runners and the Decentralized BFCL runner use the prompt contract only.

In Independent, replicas never exchange messages, so only the prompt contract applies there. Malformed reports are never rejected or re-prompted.

## The three formats

| Reader name | Code name | What each report must contain |
| --- | --- | --- |
| Freeform | `freeform` | Nothing; prompts are unchanged. This is the control. |
| Semi-structured | `semi_structured` | Five tagged sections: `[STATUS]`, `[SUMMARY]`, `[EVIDENCE_OR_TESTS]`, `[CONFIDENCE]`, `[NEXT]`, plus optional dataset tags. |
| Structured | `structured_soft` | One JSON object after `JSON_REPORT:` and before `END_JSON_REPORT`, with keys `status`, `summary`, `confidence`, `next`, `payload`, and no code fences. |

In both structured formats, `status` must be one of `not_started`, `in_progress`, `completed`, `blocked`, and `confidence` one of `low`, `medium`, `high`. A Structured report parses only if the JSON is an object with all five keys and `payload` is itself an object.

Here is one HotpotQA report in each format. The last line is the scorer-facing artifact from the output contract, unchanged across formats.

=== "Freeform"

    ```text
    Arthur's Magazine was founded in 1844; First for Women in 1989, so Arthur's
    Magazine came first.
    Answer: Arthur's Magazine
    ```

=== "Semi-structured"

    ```text
    [STATUS]
    completed
    [SUMMARY]
    Arthur's Magazine (1844) predates First for Women (1989).
    [EVIDENCE_OR_TESTS]
    Arthur's Magazine founded 1844; First for Women founded 1989.
    [CONFIDENCE]
    high, both founding years confirmed from Wikipedia.
    [NEXT]
    Use Arthur's Magazine as the final answer.
    [ANSWER_CANDIDATE]
    Arthur's Magazine

    Answer: Arthur's Magazine
    ```

=== "Structured"

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

Each dataset suggests its own optional tags and payload fields:

| Dataset | Optional Semi-structured tags | Structured `payload` fields | Final artifact |
| --- | --- | --- | --- |
| HotpotQA | `[ENTITIES]`, `[HOPS]`, `[ANSWER_CANDIDATE]` | `entities`, `hops`, `evidence`, `answer_candidate` | `Answer: <short-form>` |
| LiveCodeBench | `[APPROACH]`, `[COMPLEXITY]`, `[EDGE_CASES]`, `[CODE_STATUS]` | `approach`, `complexity`, `edge_cases`, `tests`, `code_status` | fenced `python` block |
| BFCL | `[FUNCTION_CHOICE]`, `[ARG_PLAN]`, `[CALL_CANDIDATE]` | `function_choice`, `arg_plan`, `call_candidate` | fenced `json` call list |
| ToolHop | `[TOOL_CHAIN]`, `[OBSERVATIONS]`, `[ANSWER_CANDIDATE]` | `tool_chain`, `observations`, `answer_candidate` | `<answer>...</answer>` |
| API-Bank | none; one short sentence per section, final call after `[NEXT]` | `api_choice`, `call_candidate` | one bracketed API call |
| SWE-bench | `[BUG_LOCATION]`, `[PATCH_PLAN]`, `[RISK_OR_REGRESSION]` | `bug_location`, `root_cause`, `patch_plan`, `regression_risk`, `tests_or_checks` | the repository diff |

## Coverage

Runners exist for every combination of:

- **Topologies:** `independent`, `sequential`, `centralized`, `decentralized`. Each runs the LangGraph runner of that topology.
- **Datasets:** `hotpotqa`, `lcb`, `bfcl`, `toolhop`, `apibank`, `swe`.
- **Formats:** `freeform`, `semi_structured`, `structured_soft`.

The path pattern is `communications/<topology>/<dataset>/<dataset>_<format>.py`. Each file is a few lines: it calls `install` from [`communications/communication_formats.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/communications/communication_formats.py), which loads the topology runner as a module of its own with the format preset, records its hand-offs and scores its reports.

## Run a cell

```bash title="Centralized HotpotQA with Structured messages"
python -m communications.centralized.hotpotqa.hotpotqa_structured_soft --batch --limit 100
```

All 72 runners share the runners' command line (`--batch`, `--limit`, `--offset`, `--only`, `--out-dir`, `--out`); the BFCL runners add `--category`. They have no demo, and without `--limit` they run every row. Results go to `results/communications_baseline/<topology>_<dataset>_<format>/results.jsonl` (BFCL: one subfolder per category) unless you pass `--out-dir` or `--out`, and the file is emptied first. ToolHop cells need `export TOOLHOP_ALLOW_DATASET_EXEC=1`.

Each row holds the dataset's usual score fields plus `communication_format`, the parsed reports and the rendered hand-offs (`communication_inflight_handoffs`). The parse metrics judge each agent's own text against the format: `communication_parse_rate` is the share of reports that parse as written, and `communication_all_parse_ok` is true when all do. The re-rendered reports always parse; their rate is reported separately as `communication_render_parse_rate`.

## Run the sweep

`scripts/run_communications.sh` runs each selected cell as its own process, at the default output path. Choose cells with `TOPOLOGIES`, `DATASETS` and `FORMATS` (all values by default):

```bash title="Sweep two topologies, two datasets, two formats"
TOPOLOGIES="sequential centralized" \
DATASETS="hotpotqa lcb" \
FORMATS="freeform structured_soft" \
bash scripts/run_communications.sh
```

The script runs BFCL and SWE-bench on their eval IDs (BFCL one category at a time) and the other datasets with fixed limits (HotpotQA, ToolHop and API-Bank 100, LiveCodeBench 50). It sets `TOOLHOP_ALLOW_DATASET_EXEC=1` unless you set it, and reads `VLLM_BASE_URL` and `MODEL_ID` like every runner.

## Optimize a cell

Pass `--communication <format>` to the run protocol, or use the key `<topology>_communications_<format>`. The bridge has adapters for HotpotQA, LiveCodeBench, BFCL, ToolHop and API-Bank; the experiment grid covers HotpotQA, LiveCodeBench and BFCL with GEPA, MIPRO, MAPRO and MASPO, and other cells need `--allow-any-cell`. For example, MASPO on Sequential LiveCodeBench with Structured messages:

```bash title="MASPO on Sequential · LiveCodeBench · Structured"
python -m optimizers.protocol.run --method maspo --dataset lcb \
  --topology sequential --communication structured_soft \
  --model qwen --seed 0 --out runs/maspo/lcb/sequential_structured_soft/qwen/0
```

See [Run an Optimizer](../optimizers/running.md).
