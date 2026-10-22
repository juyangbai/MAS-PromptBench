# Connect a Model

Every agent in every topology calls the same OpenAI-compatible chat endpoint. Serve the task model with vLLM, or point the runners at another server that accepts vLLM's sampling fields.
{ .lede }

## Three variables

| Variable | What it sets | Default |
| --- | --- | --- |
| `VLLM_BASE_URL` | The endpoint URL | `http://localhost:8000/v1` |
| `MODEL_ID` | The model every agent uses | `Qwen/Qwen3.5-9B` |
| `OPENAI_API_KEY` | The key, if the server needs one | `EMPTY` |

The runners read these when they start, so export them in the shell you run from.

=== "Local vLLM"

    ```bash title="Serve Qwen3.5-9B and point at it"
    bash models/serve_qwen3_5_9b.sh        # one replica per visible GPU, from port 8000

    export VLLM_BASE_URL=http://localhost:8000/v1
    export MODEL_ID=Qwen/Qwen3.5-9B
    ```

    No API key is needed for a local endpoint; the runners send `EMPTY`.

=== "Another server"

    ```bash title="Any compatible server"
    export VLLM_BASE_URL=<server-url>/v1
    export MODEL_ID=<model-name>
    export OPENAI_API_KEY=<your-key>
    ```

    The server must support tool calling, since most topologies give agents tools, and accept the vLLM fields described below.

## Decoding

The runners send one decoding protocol with each request, read from the environment:

| Setting | Value | Override |
| --- | --- | --- |
| Temperature | `0.0` | `TASK_MODEL_TEMPERATURE` |
| Top-p | `0.9` | `TASK_MODEL_TOP_P` |
| Output tokens per call | at most `32768` | `TASK_MODEL_MAX_TOKENS` |
| Request seed | `0` | `REQUEST_SEED` |
| vLLM extras | `repetition_penalty` 1.05, thinking off (`enable_thinking: false`) | none |

Independent replicas, and the debate peers of the ToolHop, API-Bank and Agents SDK runners, send their own request seeds; the Agents SDK runners leave out `repetition_penalty`. Serve a context window larger than the output cap; the 9B script serves 131,072 tokens. Optimizer rollouts sample at temperature 0.2 instead; the protocol sets that itself.

## Serve models locally

The three scripts in `models/` start vLLM's OpenAI-compatible server inside the `mas-promptbench` environment.

| Script | Model | Serving | GPUs |
| --- | --- | --- | --- |
| [`serve_qwen3_5_9b.sh`](https://github.com/juyangbai/MAS-PromptBench/blob/main/models/serve_qwen3_5_9b.sh) | `Qwen/Qwen3.5-9B` (task model) | one replica per GPU, ports 8000+ | ≥ 1 CUDA GPU |
| [`serve_llama3_1_8b.sh`](https://github.com/juyangbai/MAS-PromptBench/blob/main/models/serve_llama3_1_8b.sh) | `meta-llama/Llama-3.1-8B-Instruct` (task model) | one replica per GPU, ports 8100+ | ≥ 1 CUDA GPU |
| [`serve_qwen3_5_122b.sh`](https://github.com/juyangbai/MAS-PromptBench/blob/main/models/serve_qwen3_5_122b.sh) | `Qwen/Qwen3.5-122B-A10B-FP8` (reflection model) | tensor-parallel (TP = 4), port 8200 | 4 FP8-capable GPUs (Hopper or Blackwell) |

The two replica scripts share their settings:

| Variable | Default | Effect |
| --- | --- | --- |
| `VLLM_GPU_LIST` | all visible GPUs | Comma-separated GPU indices, one replica each |
| `VLLM_BASE_PORT` | `8000` (Qwen), `8100` (Llama) | Port of the first replica; the rest follow |
| `VLLM_HOST` | `0.0.0.0` | Bind address |
| `HF_HOME` | `$HOME/models` | Model cache |
| `MAX_MODEL_LEN` | `131072` | Context length |
| `GPU_MEMORY_UTIL` | `0.90` | vLLM memory fraction |
| `KV_CACHE_DTYPE` | `auto` | Set `fp8` on Hopper or Blackwell to roughly halve KV memory |
| `CONDA_ENV` | `mas-promptbench` | Environment the script activates |

With four GPUs, the Qwen script gives four endpoints, on ports 8000 to 8003. A runner uses one endpoint (`VLLM_BASE_URL`); an optimizer job can spread its calls across all of them. The 122B script listens on `VLLM_PORT` (default 8200). See [Environment Variables](../reference/environment.md) for every serving setting.

!!! note "Llama is gated"
    Accept the model's license on Hugging Face and export `HF_TOKEN` before the first download, or serve a downloaded copy with `HF_HUB_OFFLINE=1`.

## Endpoints for the optimizers

An optimizer job uses two models: the **task model** that runs the agents and the **reflection model** that proposes new prompts. Every one of the eight optimizers reads the same two settings:

```bash title="Task and reflection endpoints"
export TASK_ENDPOINTS=http://localhost:8000/v1             # comma-separated list is allowed
export REFLECTION_MODEL_BASE_URL=http://localhost:8200/v1  # the 122B script's port
```

`TASK_ENDPOINTS` falls back to `VLLM_BASE_URL`, and `--task-endpoints` overrides both. `REFLECTION_MODEL_BASE_URL` defaults to `http://localhost:8200/v1`. The job picks the task model with `--model qwen` or `--model llama`. The reflection model is `Qwen/Qwen3.5-122B-A10B-FP8`; another `REFLECTION_MODEL_ID` makes a job non-conformant. See [Run an Optimizer](../optimizers/running.md).

## Check the connection

```bash title="One smoke demo"
python -m topologies.single.hotpotqa.langgraph_hotpotqa
```

If the endpoint is wrong you'll see a connection error on the first model call. If it works, the demo prints an answer, its exact-match and F1 scores, and the message trace.

## Next step

<div class="cards" markdown>

- [Quick Start](quick-start.md)
  Run a baseline, optimize its prompts and read the gain.

</div>
