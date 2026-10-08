# Communication Protocols

The same topology can pass messages as free text, as tagged reports or as JSON. This page covers the three message formats, which cells use them, how to run them and how they change what prompt optimization achieves.
{ .lede }

<div class="facts" markdown>
<div><span>Formats</span>3</div>
<div><span>Topologies</span>4</div>
<div><span>Datasets</span>5</div>
<div><span>Runners</span>60</div>
</div>

## What a format changes

A format governs only the inter-agent hand-off: the report an agent passes to the next stage, the manager, a peer or the aggregator. The model, topology, roles, tools and scorer stay the same, and the scorer-facing final answer keeps its usual form after the report.

The format is applied in two places, by [`communications/communication_formats.py`](https://github.com/juyangbai/MAS-PromptBench/blob/main/communications/communication_formats.py):

1. **In the prompts.** For HotpotQA, LiveCodeBench and SWE-bench, a format contract is appended to every role's system prompt, telling the agent how to write its report and to put the final artifact after it. The ToolHop and API-Bank runners build their prompts differently and don't receive this contract.
2. **In the hand-offs.** Before a receiver reads another agent's output, the runner re-renders that text into the chosen format (`format_handoff`, or the report builder in the ToolHop and API-Bank runners). The receiver always gets a well-formed message, even when the sender ignored the contract.

In Independent, replicas never exchange messages, so only the prompt contract applies there; Independent ToolHop and API-Bank cells run the same agents under every format. Malformed reports are never rejected or re-prompted.

## The three formats

| Reader name | Code name | What each report must contain |
| --- | --- | --- |
| Freeform | `freeform` | Nothing; prompts are unchanged. This is the control. |
| Semi-structured | `semi_structured` | Five tagged sections: `[STATUS]`, `[SUMMARY]`, `[EVIDENCE_OR_TESTS]`, `[CONFIDENCE]`, `[NEXT]`, plus optional dataset tags. |
| Structured | `structured_soft` | One JSON object after `JSON_REPORT:` and before `END_JSON_REPORT`, with keys `status`, `summary`, `confidence`, `next`, `payload`, and no code fences. |

In both structured formats, `status` must be one of `not_started`, `in_progress`, `completed`, `blocked`, and `confidence` one of `low`, `medium`, `high`. A Structured report parses only if the JSON is an object with all five keys and `payload` is itself an object.

Here is one HotpotQA report in each format. The last line is the scorer-facing artifact from `topologies/output_contracts.py` (re-exported by `communications/output_contracts.py`), unchanged across formats.

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
| ToolHop | `[TOOL_CHAIN]`, `[OBSERVATIONS]`, `[ANSWER_CANDIDATE]` | `tool_chain`, `observations`, `answer_candidate` | `<answer>...</answer>` |
| API-Bank | none; one short sentence per section, final call after `[NEXT]` | `api_choice`, `call_candidate` | one bracketed API call |
| SWE-bench | `[BUG_LOCATION]`, `[PATCH_PLAN]`, `[RISK_OR_REGRESSION]` | `bug_location`, `root_cause`, `patch_plan`, `regression_risk`, `tests_or_checks` | the repository diff |

## Coverage

Runners exist for every combination of:

- **Topologies:** `independent`, `sequential`, `centralized`, `decentralized`. Each wraps the LangGraph runner of that topology.
- **Datasets:** `hotpotqa`, `lcb`, `toolhop`, `apibank`, `swe`.
- **Formats:** `freeform`, `semi_structured`, `structured_soft`.

The path pattern is `communications/<topology>/<dataset>/<dataset>_<format>.py`.

## Run a cell

```bash title="Centralized HotpotQA with Structured messages"
python -m communications.centralized.hotpotqa.hotpotqa_structured_soft --batch --limit 100
```

All 60 runners share one command line: `--batch`, `--limit`, `--offset`, `--only` (repeat it once per ID) and `--out`. Results go to `results/communications_baseline/<topology>_<dataset>_<format>/results.jsonl` unless you pass `--out <file.jsonl>`; an existing file at that path is overwritten. Each row holds the dataset's usual score fields plus `communication_format` and the rendered hand-offs (`communication_inflight_handoffs`). The parse fields (`communication_parse_rate`, `communication_all_parse_ok`) are computed on the runner's re-rendered reports, so they check the rendering, not the model's own text. ToolHop cells need `export TOOLHOP_ALLOW_DATASET_EXEC=1`.

!!! warning "SWE-bench cells"
    The shared batch loop calls each runner's `run_one(instance, None)`, but the SWE-bench topology runners also require a workdir and an output directory. Expect every SWE-bench row to be written as an error record until that call is fixed.

## Run the sweep

`scripts/run_communications.sh` runs each selected cell as its own process, at the default output path. Choose cells with `TOPOLOGIES`, `DATASETS` and `FORMATS` (all values by default):

```bash title="Sweep two topologies, two datasets, two formats"
TOPOLOGIES="sequential centralized" \
DATASETS="hotpotqa lcb" \
FORMATS="freeform structured_soft" \
bash scripts/run_communications.sh
```

The script uses fixed limits per dataset (HotpotQA, ToolHop and API-Bank 100, LiveCodeBench 50, SWE-bench 30), sets `TOOLHOP_ALLOW_DATASET_EXEC=1`, and reads `VLLM_BASE_URL` and `MODEL_ID` like every runner.

## Optimize a cell

The optimizer topology name is `<topology>_communications_<format>`, for example `centralized_communications_semi_structured`. These names exist for HotpotQA, LiveCodeBench, ToolHop and API-Bank; SWE-bench has no optimizer adapter for protocol cells.

```bash title="GEPA on Sequential / HotpotQA with Structured messages"
cd optimizers/gepa
python -m real_runner_gepa.pilots.run_gepa_dataset --dataset hotpotqa \
  --topology sequential_communications_structured_soft \
  --train-size 25 --val-size 25 --max-full-evals 5 \
  --out results/gepa/sequential_communications_structured_soft_hotpotqa
```

For Independent and Decentralized cells, add `--n-agents 4` (and `--n-rounds 2` for Decentralized) to match the baseline team; the pilot defaults are 2 agents and 1 round.
