# Vendored from MASPOB, https://github.com/HZ1008/MASPOB, commit 0be3af42da83b9b5ab187c101ec6d2ce87a5fe04,
# file scripts/prompts/generator.py. Z. Hong, Q. Zhang, J. Sun, Z. Shang, M. Kong, X. Wang, Y. Shu and Z. Dai,
# "MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks",
# arXiv:2603.02630, 2026. Upstream terms: "This project is released for research purposes. Please refer
# to the paper for details." The pinned upstream tree contains no LICENSE, COPYING or NOTICE file.
# Everything below the end-of-header line is byte-identical to the upstream file; its SHA-256 is pinned
# in ../regime.py (CORE_SOURCE_SHA256) and verified before the file is executed.
# --- end of vendoring header ---
import asyncio
import json
import os
import re
from typing import Dict, List

from scripts.prompts.prompt import (
    ITERATIVE_GENERATE_PROMPT,
    PRESET_STYLE_NAMES,
    generate_style_instruction,
)


def extract_placeholders(text: str) -> set:
    """Extract all {placeholder} patterns from text (ignores escaped {{ }})."""
    return set(re.findall(r'\{(\w+)\}', text))


def clean_generated_prompt(raw_prompt: str) -> str:
    """Strip common LLM wrapping from a generated prompt.

    Handles:
    1. Markdown code block wrappers (```...```)
    2. Explanatory preamble lines
    3. Trailing notes/comments
    """
    text = raw_prompt.strip()

    # 1. Remove markdown code blocks
    code_block_pattern = r'^```(?:\w+)?\s*\n(.*?)\n```\s*$'
    match = re.match(code_block_pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # 2. Remove common preamble lines ("Here is ...", "Below is ...")
    first_line_patterns = [
        r'^(?:Here is|Below is|The following is)[^\n]*\n+',
        r'^(?:Generated prompt|New prompt)[^\n]*:\s*\n+',
    ]
    for pattern in first_line_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 3. Remove trailing notes (e.g. "Note: ..." or "This prompt...")
    trailing_patterns = [
        r'\n+(?:Note:|This prompt|Explanation:)[^\{]*$',
    ]
    for pattern in trailing_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()


def validate_prompt_placeholders(new_prompt: str, original_prompt: str) -> bool:
    """Verify that a generated prompt preserves all original placeholders.

    Rules:
    1. All placeholders in original_prompt must appear in new_prompt.
    2. new_prompt must not introduce placeholders absent from original_prompt.
    """
    original_placeholders = extract_placeholders(original_prompt)
    new_placeholders = extract_placeholders(new_prompt)

    missing = original_placeholders - new_placeholders
    extra = new_placeholders - original_placeholders

    return len(missing) == 0 and len(extra) == 0


async def generate_prompts_from_initial(
    llm,
    initial_prompt: str,
    prompt_type: str = "UNKNOWN",
    prompt_goal: str = "Achieve the same goal as the original prompt",
    num_prompts: int = 100,
    dataset: str = None,
) -> List[str]:
    """Generate prompt variants from an initial template using iterative style sampling.

    Logic:
    1. The first prompt is always initial_prompt (baseline).
    2. Remaining prompts are style-varied versions of the template.
    3. Placeholder integrity is validated for each variant.

    Randomness source: generate_style_instruction(random_sample=True)
    with 10 dimensions × 10 options = 10^10 possible combinations.
    """
    print(f"[Prompt] Generating {num_prompts} variants for {prompt_type}")
    print(f"  Goal: {prompt_goal[:80]}...")
    print(f"  Mode: Dynamic style generation (first = initial template, rest = variants)")
    if dataset:
        print(f"  Dataset: {dataset}")

    required_placeholders = extract_placeholders(initial_prompt)
    print(f"  Required placeholders: {required_placeholders}")

    # First prompt is always the initial template
    prompts = [initial_prompt.strip()]
    print(f"  [#0] Initial template (baseline)")

    validation_failures = 0

    placeholder_list = ", ".join(f"{{{p}}}" for p in sorted(required_placeholders))

    # Generate num_prompts - 1 variants in parallel (first slot is the initial template)
    remaining_count = num_prompts - 1
    if remaining_count <= 0:
        print(f"  Only initial template required; skipping variant generation.")
        return prompts

    batch_size = 20
    target_count = int(remaining_count * 1.5)  # over-generate to compensate for validation failures

    async def generate_single_prompt(meta_prompt: str, style_type: str) -> tuple:
        """Generate a single prompt; returns (prompt, style_type) or (None, style_type) on failure."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                raw_prompt = await llm(meta_prompt)
                new_prompt = clean_generated_prompt(raw_prompt)
                if validate_prompt_placeholders(new_prompt, initial_prompt):
                    return (new_prompt, style_type)
            except Exception:
                await asyncio.sleep(0.5 * (attempt + 1))
        return (None, style_type)

    print(f"  [Parallel] Generating {target_count} variant candidates in batches of {batch_size}...")

    all_meta_prompts = []
    all_style_types = []

    # Prioritize dataset-specific presets, then fill with random sampling
    dataset_lower = (dataset or "").lower()
    if dataset_lower in ["drop", "hotpotqa"]:
        # F1-scored tasks: prefer concise styles
        priority_presets = ["CONCISE_DIRECT", "QUICK_CONCISE", "SYSTEMATIC"]
    elif dataset_lower in ["gsm8k", "math"]:
        # Math tasks: prefer detailed reasoning styles
        priority_presets = ["DETAILED_REASONING", "VERIFY_FIRST", "SYSTEMATIC"]
    elif dataset_lower in ["humaneval", "mbpp"]:
        # Code tasks: prefer rigorous code styles
        priority_presets = ["CODE_RIGOROUS", "SYSTEMATIC", "PATTERN_BASED"]
    else:
        priority_presets = []

    other_presets = [p for p in PRESET_STYLE_NAMES if p not in priority_presets]
    all_presets = priority_presets + other_presets

    for i in range(target_count):
        if i < len(all_presets):
            # Use preset styles first for guaranteed diversity
            preset = all_presets[i]
            style_instruction = generate_style_instruction(quality=preset)
            style_type = f"PRESET:{preset}"
        else:
            # Fall back to random sampling after exhausting presets
            style_instruction = generate_style_instruction(random_sample=True)
            style_type = "RANDOM"

        meta_prompt = ITERATIVE_GENERATE_PROMPT.format(
            prompt_type=prompt_type,
            prompt_goal=prompt_goal,
            required_placeholders=placeholder_list,
            style_instruction=style_instruction,
        )
        all_meta_prompts.append(meta_prompt)
        all_style_types.append(style_type)

    # Execute in parallel batches
    for batch_start in range(0, len(all_meta_prompts), batch_size):
        if len(prompts) >= num_prompts:
            break

        batch_end = min(batch_start + batch_size, len(all_meta_prompts))
        batch_meta = all_meta_prompts[batch_start:batch_end]
        batch_styles = all_style_types[batch_start:batch_end]

        tasks = [generate_single_prompt(mp, st) for mp, st in zip(batch_meta, batch_styles)]
        results = await asyncio.gather(*tasks)

        for new_prompt, style_type in results:
            if new_prompt is not None and len(prompts) < num_prompts:
                prompts.append(new_prompt)
            elif new_prompt is None:
                validation_failures += 1

        print(f"  [Batch {batch_start//batch_size + 1}] {len(prompts)}/{num_prompts} valid prompts (1 initial + {len(prompts)-1} variants)")

    if len(prompts) < num_prompts:
        print(f"\n  [Warning] Only generated {len(prompts)}/{num_prompts} prompts")

    print(f"  Generated: {len(prompts)} prompts (1 initial template + {len(prompts)-1} variants, validation failures: {validation_failures})")
    return prompts


async def get_or_create_prompt_domain(
    llm, prompt_name: str, initial_prompt: str,
    prompt_goal: str = "Achieve the same goal as the original prompt",
    num_prompts: int = 20, domain_dir: str = "prompt_domain",
    dataset: str = None,
) -> List[str]:
    """Load or incrementally generate a prompt domain.

    Args:
        llm: LLM instance.
        prompt_name: Name of the prompt (used as filename).
        initial_prompt: Base prompt template.
        prompt_goal: Description of the prompt objective.
        num_prompts: Target number of prompts.
        domain_dir: Storage directory.
        dataset: Dataset name (used for style selection).

    Logic:
        1. If the cache file has enough prompts, load and return directly.
        2. If the cache exists but is insufficient, generate only the missing prompts.
        3. If no cache exists, generate all prompts from scratch.
    """
    os.makedirs(domain_dir, exist_ok=True)
    file_path = os.path.join(domain_dir, f"{prompt_name}.txt")

    existing_prompts = []

    # Try loading cached prompts
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            try:
                data = json.loads(content)
                if isinstance(data, list) and data:
                    existing_prompts = data
                    if len(existing_prompts) >= num_prompts:
                        print(f"[Cache] {prompt_name}: loaded {num_prompts}/{len(existing_prompts)} prompts (skipped)")
                        return existing_prompts[:num_prompts]
                    else:
                        print(f"[Cache] {prompt_name}: have {len(existing_prompts)}, need {num_prompts}, generating {num_prompts - len(existing_prompts)} more...")
            except json.JSONDecodeError:
                # Malformed file: treat content as the first prompt
                existing_prompts = [content]

    needed = num_prompts - len(existing_prompts)

    if needed <= 0:
        return existing_prompts[:num_prompts]

    # Incremental generation: only generate what's missing
    new_prompts = await generate_prompts_from_initial(
        llm, initial_prompt, prompt_name, prompt_goal,
        num_prompts=needed + 1,  # +1 because index 0 is the initial template
        dataset=dataset
    )

    # Merge existing and new prompts (deduplicated)
    if existing_prompts:
        combined = existing_prompts + [p for p in new_prompts[1:] if p not in existing_prompts]
    else:
        combined = new_prompts

    final_prompts = combined[:num_prompts]

    # Save the full list back to cache
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(final_prompts, f, ensure_ascii=False, indent=2)

    print(f"[Cache] {prompt_name}: saved {len(final_prompts)} prompts ({len(existing_prompts)} existing + {len(final_prompts) - len(existing_prompts)} new)")
    return final_prompts


async def build_prompt_domains(
    llm, prompt_configs: Dict[str, Dict],
    num_prompts: int = 100, domain_dir: str = "prompt_domain",
    dataset: str = None,
) -> Dict[str, List[str]]:
    """Build or load all prompt domains in bulk.

    Args:
        llm: LLM instance.
        prompt_configs: Dict mapping prompt names to their config dicts.
        num_prompts: Number of prompts to generate per type.
        domain_dir: Storage directory.
        dataset: Dataset name (used for style selection).
    """
    result = {}
    for name, cfg in prompt_configs.items():
        result[name] = await get_or_create_prompt_domain(
            llm, name, cfg["template"],
            cfg.get("goal", "Achieve the same goal as the original prompt"),
            num_prompts, domain_dir, dataset=dataset,
        )
    return result

