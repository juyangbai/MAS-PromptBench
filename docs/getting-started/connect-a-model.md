# Connect a Model

Every agent in every topology calls the same OpenAI-compatible chat endpoint. Point it at a hosted API, or serve Qwen yourself with vLLM.
{ .lede }

## Three variables

| Variable | What it sets | Runner default |
| --- | --- | --- |
| `VLLM_BASE_URL` | The endpoint URL, hosted or local | a lab-internal host; always set it |
| `MODEL_ID` | The model every agent uses | `Qwen/Qwen3.5-9B` |
| `OPENAI_API_KEY` | The key, if your provider needs one | `EMPTY` |

The runners read these when they start, so export them in the shell you run from. The built-in `VLLM_BASE_URL` defaults differ between runners (`http://localhost:8000/v1` or `http://localhost:8001/v1`), so set it explicitly.

=== "Hosted API (no GPU)"

    ```bash title="Any OpenAI-compatible provider"
    export VLLM_BASE_URL=https://api.openai.com/v1
    export MODEL_ID=<model-name>
    export OPENAI_API_KEY=<your-key>
    ```

    Any provider that speaks the OpenAI chat-completions API works, as long as the model supports tool calling: most topologies give agents tools.

=== "Local vLLM"

    ```bash title="Serve Qwen3.5-9B and point at it"
    bash models/serve_qwen3_5_9b.sh        # one replica per visible GPU, from port 8000

    export VLLM_BASE_URL=http://localhost:8000/v1
    export MODEL_ID=Qwen/Qwen3.5-9B
    ```

    No API key is needed for a local endpoint; the runners send `EMPTY`.

!!! warning "The repository has no `.env` file"
    The README mentions a `.env` file with provider blocks, but none is included. Export the variables directly as shown, or keep them in your own `.env` and load it with `set -a && source .env && set +a`.

## Serve models locally

The two scripts in `models/` start vLLM's OpenAI-compatible server inside the `mas-promptbench` environment.

| Script | Model | Serving | GPUs |
| --- | --- | --- | --- |
| [`serve_qwen3_5_9b.sh`](https://github.com/juyangbai/MAS-PromptBench/blob/main/models/serve_qwen3_5_9b.sh) | `Qwen/Qwen3.5-9B` | one replica per GPU, consecutive ports | ≥ 1 CUDA GPU |
| [`serve_qwen3_5_122b.sh`](https://github.com/juyangbai/MAS-PromptBench/blob/main/models/serve_qwen3_5_122b.sh) | `Qwen/Qwen3.5-122B-A10B-FP8` | tensor-parallel, TP = 4 | 4 FP8-capable GPUs (Hopper or Blackwell) |

The 9B script is configured through environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `VLLM_GPU_LIST` | all visible GPUs | Comma-separated GPU indices, one replica each |
| `VLLM_BASE_PORT` | `8000` | Port of the first replica; the rest follow |
| `VLLM_HOST` | `0.0.0.0` | Bind address |
| `HF_HOME` | `$HOME/models` | Model cache |
| `MAX_MODEL_LEN` | `131072` | Context length |
| `GPU_MEMORY_UTIL` | `0.90` | vLLM memory fraction |
| `KV_CACHE_DTYPE` | `auto` | Set `fp8` on Hopper or Blackwell to roughly halve KV memory |
| `CONDA_ENV` | `mas-promptbench` | Environment the script activates |

With four GPUs you get four endpoints, on ports 8000 to 8003. The runners use one endpoint (`VLLM_BASE_URL`); the [optimizers](../optimizers/running.md) can spread calls across all of them.

## Endpoints for the optimizers

GEPA and MIPRO don't read `VLLM_BASE_URL`. They take their own variables, and they use two models: a **task model** that runs the agents and a **reflection model** (GEPA) or **proposal model** (MIPRO) that writes new prompts.

```bash title="Reuse one endpoint for both roles"
export GEPA_TASK_ENDPOINTS=$VLLM_BASE_URL     # comma-separated list is allowed
export GEPA_REFL_ENDPOINT=$VLLM_BASE_URL
export TASK_MODEL=$MODEL_ID REFL_MODEL=$MODEL_ID
```

MIPRO reads `MIPRO_TASK_ENDPOINTS` and `MIPRO_REFL_ENDPOINT` and falls back to the GEPA names. If you leave these unset, the optimizers send requests to built-in defaults (a fixed list of local ports) and fail unless your servers listen there. The code's defaults pair Qwen3.5-9B agents with Qwen3.5-122B for reflection. See [Environment Variables](../reference/environment.md) for the full list.

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
