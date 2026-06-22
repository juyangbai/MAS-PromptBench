"""Generate initial role system prompts for every (topology, dataset, role)."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import yaml
from openai import OpenAI


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

def strip_thinking(text: str) -> str:
    """Remove Qwen3-style reasoning from a model reply.

    Qwen3's vLLM chat template injects the opening <think> as part of the
    assistant prefix, so the model's response begins with reasoning (no
    opening tag) and ends that block with </think> before the final answer.
    Cut everything up through the last </think>.
    """
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>"):]
    return text.strip()


def render(template: str, **variables: str) -> str:
    output = template
    for key, value in variables.items():
        output = output.replace("{" + key + "}", value.strip())
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    # Prompt generation uses the 122B model by default: role prompts are
    # written once per (topology, benchmark, role) and benefit from a
    # stronger model's instruction-following. Dataset runtime uses the
    # 9B via VLLM_BASE_URL / MODEL_ID; we deliberately DON'T read those
    # here so dataset env vars can't silently flip the generator onto a
    # smaller model. Override the generator-side defaults with
    # PROMPT_GEN_BASE_URL / PROMPT_GEN_MODEL if needed.
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PROMPT_GEN_BASE_URL", "http://lai:8000/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PROMPT_GEN_API_KEY", "EMPTY"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PROMPT_GEN_MODEL", "Qwen/Qwen3.5-122B-A10B-FP8"),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Restrict to <topology> or <topology>/<dataset> or <topology>/<dataset>/<role>.",
    )
    args = parser.parse_args()

    meta_prompt = (PROMPTS_DIR / "meta_prompt.txt").read_text()
    domains = yaml.safe_load((PROMPTS_DIR / "domains.yaml").read_text())
    roles_config = yaml.safe_load((PROMPTS_DIR / "roles.yaml").read_text())
    tools_config = yaml.safe_load((PROMPTS_DIR / "tools.yaml").read_text())

    filters = [tuple(item.split("/")) for item in (args.only or [])]

    def _prefix_match(parts: tuple[str, ...], filt: tuple[str, ...]) -> bool:
        n = min(len(parts), len(filt))
        return parts[:n] == filt[:n]

    def _may_match(*parts: str) -> bool:
        return not filters or any(_prefix_match(parts, filt) for filt in filters)

    def _selected(*parts: str) -> bool:
        return not filters or any(parts[: len(filt)] == filt for filt in filters)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    total_count = finish_count = skipped_count = failed_count = 0
    run_start_time = time.time()

    print(
        f"generator: model={args.model}  base_url={args.base_url}  "
        f"temperature={args.temperature}  seed={args.seed}",
        file=sys.stderr,
    )

    for topology, topology_config in roles_config["topologies"].items():
        if not _may_match(topology):
            continue
        topology_description = topology_config["description"]
        # Role config is benchmark-specific: roles.yaml nests roles under
        # topology -> benchmarks -> <benchmark> -> <role>: <description>
        benchmarks = topology_config.get("benchmarks", {})
        for dataset, domain_description in domains.items():
            if not _may_match(topology, dataset):
                continue
            dataset_roles = benchmarks.get(dataset)
            if not dataset_roles:
                # Topology doesn't define roles for this benchmark (either
                # unsupported or not yet filled in); skip silently.
                continue
            cell_tools = tools_config.get(topology, {}).get(dataset, "None.")
            for role, role_description in dataset_roles.items():
                if not _selected(topology, dataset, role):
                    continue
                total_count += 1
                output_path = PROMPTS_DIR / topology / dataset / f"{role}.txt"
                if output_path.exists() and not args.force:
                    skipped_count += 1
                    print(f"[skip]  {topology}/{dataset}/{role}", file=sys.stderr)
                    continue

                user_message = render(
                    meta_prompt,
                    DOMAIN_BACKGROUND=domain_description,
                    TOPOLOGY_DESC=topology_description,
                    TOOLS=cell_tools,
                    ROLE=role,
                    ROLE_DESC=role_description,
                )

                call_start_time = time.time()
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": user_message}],
                        temperature=args.temperature,
                        seed=args.seed,
                    )
                    generated_prompt = strip_thinking(response.choices[0].message.content)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(generated_prompt + "\n")
                    finish_count += 1
                    elapsed_seconds = time.time() - call_start_time
                    print(f"[finish] {topology}/{dataset}/{role}  ({elapsed_seconds:.1f}s)", file=sys.stderr)
                except Exception as error:
                    failed_count += 1
                    print(f"[fail]  {topology}/{dataset}/{role}  {error}", file=sys.stderr)

    total_elapsed = time.time() - run_start_time
    print(
        f"\ndone: {finish_count}/{total_count} finish, {skipped_count} skipped, {failed_count} failed  [{total_elapsed:.0f}s]",
        file=sys.stderr,
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
