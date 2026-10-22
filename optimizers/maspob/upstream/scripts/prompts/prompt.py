# Vendored from MASPOB, https://github.com/HZ1008/MASPOB, commit 0be3af42da83b9b5ab187c101ec6d2ce87a5fe04,
# file scripts/prompts/prompt.py. Z. Hong, Q. Zhang, J. Sun, Z. Shang, M. Kong, X. Wang, Y. Shu and Z. Dai,
# "MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks",
# arXiv:2603.02630, 2026. Upstream terms: "This project is released for research purposes. Please refer
# to the paper for details." The pinned upstream tree contains no LICENSE, COPYING or NOTICE file.
# Everything below the end-of-header line is byte-identical to the upstream file; its SHA-256 is pinned
# in ../regime.py (CORE_SOURCE_SHA256) and verified before the file is executed.
# --- end of vendoring header ---
# =============================================================================
# ENSEMBLE PROMPT TYPES
# =============================================================================

ENSEMBLE_PROMPT_TYPES = [
    "SC_ENSEMBLE_PROMPT",
    "MD_ENSEMBLE_PROMPT",
    "REFLECTION_ON_PUBLIC_TEST_PROMPT",
]

# =============================================================================
# OPERATOR PROMPTS
# =============================================================================

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
"""

FORMAT_PROMPT = """
For the question described as {problem_description},
please extract a short and concise answer contains only one word/few words from the following solution: {solution}.
Make sure there are no additional comments or explanations in your response.
"""

# =============================================================================
# DROP dataset Prompt
# =============================================================================

SOLVE_PROMPT = """
Read the passage and question carefully. Calculate or identify the specific numerical answer requested.
Provide a clear step-by-step solution, and ensure your final answer is a single number or short phrase without any additional text.

For example:
If asked "How many years between 1990 and 2000?", respond with:
1990 to 2000 is a span of:
2000 - 1990 = 10 years
10

{input}
"""

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

FORMAT_ANSWER_PROMPT = """
Given the question and the best answer below, extract and format the final answer.
The answer should be concise - typically just a name, number, phrase, or short statement.
Do not include explanations, reasoning, or prefixes like "The answer is".
Put your formatted answer in the "answer" field.

{input}
"""



PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` with NO PARAMETERS: `def solve():`. The function must not require any input parameters. All necessary values should be hardcoded inside the function based on the problem description.
3. The `solve` function must return the final calculation result.

IMPORTANT: The function signature MUST be exactly `def solve():` with no arguments. Example:
```python
def solve():
    # Extract values from problem
    value1 = 10
    value2 = 5
    result = value1 + value2
    return result
```

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""


REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a python code solution which failed to pass test or execute, you need to carefully analyze the reason for the failure and propose a corrected code solution.

### Problem Description
{problem}

### Current Code Solution
{solution}

### Execution Result
{exec_pass}

### Failed Test Case
{test_fail}

## Analysis Instructions
Please follow these steps to debug and fix the code:

1. **Understand the Expected Behavior**: Re-read the problem description carefully. Pay special attention to:
   - Edge cases mentioned in examples (e.g., negative numbers, empty inputs, boundary conditions)
   - The exact definition of terms used (e.g., "digit sum" for negative numbers might have special handling)
   - Return type and format requirements

2. **Analyze the Test Failure**: Compare the expected output with actual output from the failed test case:
   - What input caused the failure?
   - What was the expected output vs actual output?
   - Trace through your code logic with this specific input

3. **Identify the Root Cause**: Common issues include:
   - Misunderstanding the problem requirements (e.g., per-row calculation vs total)
   - Off-by-one errors or boundary condition handling
   - Incorrect handling of negative numbers or special cases
   - Using wrong comparison operators (< vs <=, > vs >=)
   - Logic errors in conditionals

4. **Fix the Code**: Make the minimal necessary changes to fix the identified issue.

Provide ONLY the corrected Python code solution. Do not include any explanations, test cases, or additional text.
"""

MD_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the solution that is more capable of solving the problem compared to other solutions, as this is crucial for problem-solving.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

REVIEW_PROMPT = """
Given a problem and a thoughtful solution, your task is to carefully analyze and verify the solution's correctness and provide a review result in boolean format.

problem: {problem}
solution: {solution}

If you are more than 95 percent confident that the final answer is incorrect, please return False and give a feedback for the error. Otherwise, please return True and give a explanation for the correctness.
"""

REVISE_PROMPT = """
Given a problem and a thoughtful solution which is just reviewed as incorrect, your task is to revise the solution to solve the question and ensure the final code solution is wrapped with ```python```.

problem: {problem}
solution: {solution}
feedback: {feedback}

Ensure the output code is self-contained, and without any additional text or test cases.
"""

# =============================================================================
# MATH dataset Prompts
# =============================================================================

REFINE_ANSWER_PROMPT = """
Given the mathematical problem and the output from the code execution, please provide
a well-formatted and detailed solution. Follow these guidelines:
1. Begin with a clear statement of the problem.
2. Explain the approach and any formulas or concepts used.
3. Show step-by-step calculations, using LaTeX notation for mathematical expressions.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer, enclosed in \\boxed{{}} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.
Your response should be comprehensive, mathematically rigorous, and easy to follow.

Problem and code output:
{input}
"""

GENERATE_SOLUTION_PROMPT = """
Please solve the given mathematical problem step by step. Follow these guidelines:
1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Present the final answer enclosed in \\boxed{{}} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.
Your solution should be thorough, mathematically sound, and easy to understand.

Problem:
{input}
"""

DETAILED_SOLUTION_PROMPT = """
Provide a comprehensive, step-by-step solution to the given mathematical problem. Your
response should include:
1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \\boxed{{}} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.
Ensure your solution is rigorous, easy to follow, and educational for someone learning
the concept.

Problem:
{input}
"""


# =============================================================================
# Iterative Prompt Generation Template
# =============================================================================

ITERATIVE_GENERATE_PROMPT = """
Generate an instruction prompt for: {prompt_type}
Goal: {prompt_goal}

STYLE TO FOLLOW:
{style_instruction}

RULES:
1. Use EXACTLY these placeholders (no changes): {required_placeholders}
2. VARY structure: different openings, formats (bullets/steps/paragraphs), lengths
3. Output ONLY the prompt text, no explanations or markdown
"""


# =============================================================================
# Multi-dimensional Prompt Style System
# =============================================================================
#
# Design principles:
# - Each dimension's options produce meaningfully different strategy effects.
# - Options account for each dataset's evaluation characteristics.
# - Combined prompts are syntactically valid; differences are strategic, not syntactic.
#
# Dataset evaluation notes:
# - DROP/HotpotQA: F1 scoring, penalizes verbosity, needs short answers.
# - GSM8K: extracts last number; detailed reasoning helps.
# - MATH: extracts \boxed{}, requires rigorous derivation.
# - HumanEval/MBPP: code execution, requires edge-case handling.

STYLE_DIMENSIONS = {
    # =========================================================================
    # Dim 1: Reasoning style
    # =========================================================================
    "reasoning_style": [
        "Think step by step.",
        "Break down the problem systematically.",
        "Analyze each part carefully.",
        "Reason through logically.",
        "Consider all aspects before answering.",
        "Work through the problem methodically.",
        "Examine the details thoroughly.",
        "Process the information step by step.",
        "Apply logical reasoning.",
        "Evaluate systematically.",
    ],

    # =========================================================================
    # Dim 2: Output format
    # =========================================================================
    "output_format": [
        "Give only the final answer, no explanation needed.",
        "End your response with the answer on a new line.",
        "Put your final answer after 'Answer:'.",
        "State the answer directly and concisely.",
        "Conclude with just the answer.",
        "Your last line should be only the answer.",
        "After reasoning, give a one-line final answer.",
        "Provide the answer in the simplest form.",
        "Output the answer without extra words.",
        "End with: 'The answer is [your answer]'.",
    ],

    # =========================================================================
    # Dim 3: Reasoning depth
    # =========================================================================
    "reasoning_depth": [
        "Think through each step carefully before answering.",
        "Consider multiple approaches before deciding.",
        "Verify your reasoning at each step.",
        "Check for edge cases and special conditions.",
        "Make sure your logic is sound before concluding.",
        "Consider what could go wrong with your approach.",
        "Test your answer mentally before responding.",
        "Think about whether your answer makes sense.",
        "Double-check calculations and logic.",
        "Ensure consistency in your reasoning.",
    ],

    # =========================================================================
    # Dim 4: Verification requirements
    # =========================================================================
    "verification": [
        "Double-check your answer before responding.",
        "Verify your answer is correct.",
        "Review your reasoning before answering.",
        "Confirm your answer is accurate.",
        "Check your work before responding.",
        "Validate your conclusion.",
        "Ensure your answer is correct.",
        "Cross-check your reasoning.",
        "Make sure your answer is right.",
        "Verify your logic before answering.",
    ],

    # =========================================================================
    # Dim 5: Task understanding
    # =========================================================================
    "task_understanding": [
        "Read the question carefully before answering.",
        "Understand the question fully first.",
        "Analyze what is being asked.",
        "Identify the key requirements.",
        "Parse the question thoroughly.",
        "Comprehend the question completely.",
        "Understand the task before responding.",
        "Grasp the question requirements.",
        "Analyze the question carefully.",
        "Consider what is being asked.",
    ],

    # =========================================================================
    # Dim 6: Problem decomposition
    # =========================================================================
    "decomposition": [
        "Break the problem into smaller parts.",
        "Identify the key components first.",
        "List what you know and what you need to find.",
        "Separate the problem into steps.",
        "Find the core question to answer.",
        "Identify constraints and requirements.",
        "Determine what information is given.",
        "Figure out what the question is really asking.",
        "Identify the main goal and sub-goals.",
        "Clarify the problem before solving.",
    ],

    # =========================================================================
    # Dim 7: Error handling
    # =========================================================================
    "error_handling": [
        "Watch out for common mistakes.",
        "Be careful with edge cases.",
        "Check for off-by-one errors.",
        "Consider boundary conditions.",
        "Avoid assumptions not stated in the problem.",
        "Handle special cases properly.",
        "Don't overlook negative numbers or zeros.",
        "Consider what happens at limits.",
        "Be careful with order of operations.",
        "Watch for division by zero or overflow.",
    ],

    # =========================================================================
    # Dim 8: Answer validation
    # =========================================================================
    "answer_validation": [
        "Check if your answer makes sense.",
        "Verify by substituting back.",
        "Test with a simple example.",
        "Make sure units are correct.",
        "Check the answer's magnitude is reasonable.",
        "Verify the answer satisfies all conditions.",
        "Double-check critical calculations.",
        "Ensure the answer type matches what's asked.",
        "Confirm the answer is complete.",
        "Check for arithmetic errors.",
    ],

    # =========================================================================
    # Dim 9: Solving strategy
    # =========================================================================
    "solving_strategy": [
        "Try working backwards from the answer.",
        "Look for patterns in the problem.",
        "Consider similar problems you've solved.",
        "Try a simpler version first.",
        "Use examples to guide your thinking.",
        "Think about what method applies here.",
        "Consider multiple solution approaches.",
        "Use elimination to narrow down options.",
        "Draw a diagram if helpful.",
        "Organize information systematically.",
    ],

    # =========================================================================
    # Dim 10: Output conciseness
    # =========================================================================
    "output_length": [
        "Give a complete but concise answer.",
        "Include only essential information.",
        "Be thorough but not verbose.",
        "Focus on what's asked, skip the rest.",
        "Provide enough detail to be clear.",
        "Keep explanations brief.",
        "Include key steps only.",
        "Be precise, not wordy.",
        "Answer fully but efficiently.",
        "Give the minimum needed for a complete answer.",
    ],
}

# =============================================================================
# Predefined style presets
# =============================================================================
# Dimension index range: 0-9
# Dimension order: reasoning_style, output_format, reasoning_depth, verification,
#                  task_understanding, decomposition, error_handling, answer_validation,
#                  solving_strategy, output_length

QUALITY_PRESETS = {
    # Concise & direct - suited for DROP/HotpotQA (F1 scoring, penalizes verbosity)
    "CONCISE_DIRECT": {
        "reasoning_style": 3,      # "Reason through logically."
        "output_format": 0,        # "Give only the final answer, no explanation needed."
        "reasoning_depth": 7,      # "Think about whether your answer makes sense."
        "verification": 0,         # "Double-check your answer before responding."
        "task_understanding": 2,   # "Analyze what is being asked."
        "decomposition": 4,        # "Find the core question to answer."
        "error_handling": 4,       # "Avoid assumptions not stated in the problem."
        "answer_validation": 0,    # "Check if your answer makes sense."
        "solving_strategy": 7,     # "Use elimination to narrow down options."
        "output_length": 3,        # "Focus on what's asked, skip the rest."
    },
    # Detailed reasoning - suited for GSM8K/MATH (rigorous derivation)
    "DETAILED_REASONING": {
        "reasoning_style": 0,      # "Think step by step."
        "output_format": 6,        # "After reasoning, give a one-line final answer."
        "reasoning_depth": 2,      # "Verify your reasoning at each step."
        "verification": 5,         # "Validate your conclusion."
        "task_understanding": 0,   # "Read the question carefully before answering."
        "decomposition": 0,        # "Break the problem into smaller parts."
        "error_handling": 8,       # "Be careful with order of operations."
        "answer_validation": 1,    # "Verify by substituting back."
        "solving_strategy": 0,     # "Try working backwards from the answer."
        "output_length": 0,        # "Give a complete but concise answer."
    },
    # Code-rigorous - suited for HumanEval/MBPP (code execution, edge cases)
    "CODE_RIGOROUS": {
        "reasoning_style": 1,      # "Break down the problem systematically."
        "output_format": 3,        # "State the answer directly and concisely."
        "reasoning_depth": 3,      # "Check for edge cases and special conditions."
        "verification": 4,         # "Check your work before responding."
        "task_understanding": 3,   # "Identify the key requirements."
        "decomposition": 5,        # "Identify constraints and requirements."
        "error_handling": 1,       # "Be careful with edge cases."
        "answer_validation": 5,    # "Verify the answer satisfies all conditions."
        "solving_strategy": 2,     # "Consider similar problems you've solved."
        "output_length": 6,        # "Include key steps only."
    },
    # Systematic analysis
    "SYSTEMATIC": {
        "reasoning_style": 5,      # "Work through the problem methodically."
        "output_format": 2,        # "Put your final answer after 'Answer:'."
        "reasoning_depth": 0,      # "Think through each step carefully before answering."
        "verification": 2,         # "Review your reasoning before answering."
        "task_understanding": 5,   # "Comprehend the question completely."
        "decomposition": 3,        # "Separate the problem into steps."
        "error_handling": 0,       # "Watch out for common mistakes."
        "answer_validation": 6,    # "Double-check critical calculations."
        "solving_strategy": 9,     # "Organize information systematically."
        "output_length": 2,        # "Be thorough but not verbose."
    },
    # Pattern-based
    "PATTERN_BASED": {
        "reasoning_style": 2,      # "Analyze each part carefully."
        "output_format": 5,        # "Your last line should be only the answer."
        "reasoning_depth": 1,      # "Consider multiple approaches before deciding."
        "verification": 7,         # "Cross-check your reasoning."
        "task_understanding": 6,   # "Understand the task before responding."
        "decomposition": 1,        # "Identify the key components first."
        "error_handling": 5,       # "Handle special cases properly."
        "answer_validation": 2,    # "Test with a simple example."
        "solving_strategy": 1,     # "Look for patterns in the problem."
        "output_length": 4,        # "Provide enough detail to be clear."
    },
    # Verification-first
    "VERIFY_FIRST": {
        "reasoning_style": 4,      # "Consider all aspects before answering."
        "output_format": 9,        # "End with: 'The answer is [your answer]'."
        "reasoning_depth": 8,      # "Double-check calculations and logic."
        "verification": 1,         # "Verify your answer is correct."
        "task_understanding": 1,   # "Understand the question fully first."
        "decomposition": 2,        # "List what you know and what you need to find."
        "error_handling": 2,       # "Check for off-by-one errors."
        "answer_validation": 4,    # "Check the answer's magnitude is reasonable."
        "solving_strategy": 4,     # "Use examples to guide your thinking."
        "output_length": 1,        # "Include only essential information."
    },
    # Quick & concise
    "QUICK_CONCISE": {
        "reasoning_style": 8,      # "Apply logical reasoning."
        "output_format": 8,        # "Output the answer without extra words."
        "reasoning_depth": 6,      # "Test your answer mentally before responding."
        "verification": 8,         # "Make sure your answer is right."
        "task_understanding": 8,   # "Analyze the question carefully."
        "decomposition": 7,        # "Figure out what the question is really asking."
        "error_handling": 3,       # "Consider boundary conditions."
        "answer_validation": 7,    # "Ensure the answer type matches what's asked."
        "solving_strategy": 5,     # "Think about what method applies here."
        "output_length": 9,        # "Give the minimum needed for a complete answer."
    },
    # Comprehensive
    "COMPREHENSIVE": {
        "reasoning_style": 6,      # "Examine the details thoroughly."
        "output_format": 1,        # "End your response with the answer on a new line."
        "reasoning_depth": 4,      # "Make sure your logic is sound before concluding."
        "verification": 3,         # "Confirm your answer is accurate."
        "task_understanding": 4,   # "Parse the question thoroughly."
        "decomposition": 8,        # "Identify the main goal and sub-goals."
        "error_handling": 6,       # "Don't overlook negative numbers or zeros."
        "answer_validation": 3,    # "Make sure units are correct."
        "solving_strategy": 6,     # "Consider multiple solution approaches."
        "output_length": 5,        # "Keep explanations brief."
    },
    # Simplify-first
    "SIMPLIFY_FIRST": {
        "reasoning_style": 7,      # "Process the information step by step."
        "output_format": 4,        # "Conclude with just the answer."
        "reasoning_depth": 5,      # "Consider what could go wrong with your approach."
        "verification": 6,         # "Ensure your answer is correct."
        "task_understanding": 7,   # "Grasp the question requirements."
        "decomposition": 6,        # "Determine what information is given."
        "error_handling": 7,       # "Consider what happens at limits."
        "answer_validation": 8,    # "Confirm the answer is complete."
        "solving_strategy": 3,     # "Try a simpler version first."
        "output_length": 7,        # "Be precise, not wordy."
    },
    # Visual approach
    "VISUAL_APPROACH": {
        "reasoning_style": 9,      # "Evaluate systematically."
        "output_format": 7,        # "Provide the answer in the simplest form."
        "reasoning_depth": 9,      # "Ensure consistency in your reasoning."
        "verification": 9,         # "Verify your logic before answering."
        "task_understanding": 9,   # "Consider what is being asked."
        "decomposition": 9,        # "Clarify the problem before solving."
        "error_handling": 9,       # "Watch for division by zero or overflow."
        "answer_validation": 9,    # "Check for arithmetic errors."
        "solving_strategy": 8,     # "Draw a diagram if helpful."
        "output_length": 8,        # "Answer fully but efficiently."
    },
}


def generate_style_instruction(
    quality: str = None,
    dimensions: dict = None,
    random_sample: bool = False
) -> str:
    """Generate a style instruction string via preset or random sampling.

    Args:
        quality: Preset name (e.g. "CONCISE_DIRECT", "DETAILED_REASONING").
        dimensions: Manual dimension-index mapping {dim_name: index}.
        random_sample: If True, each dimension is sampled independently at random.

    Returns:
        Combined style instruction string.
    """
    import random as rand_module

    selected = {}

    if quality and quality in QUALITY_PRESETS:
        # Use preset: each dimension has a specific index configuration
        preset = QUALITY_PRESETS[quality]
        for dim, options in STYLE_DIMENSIONS.items():
            idx = preset.get(dim, 0) % len(options)
            selected[dim] = options[idx]
    elif dimensions:
        # Manually specified dimension indices
        for dim, options in STYLE_DIMENSIONS.items():
            if dim in dimensions:
                idx = dimensions[dim] % len(options)
                selected[dim] = options[idx]
            elif random_sample:
                selected[dim] = rand_module.choice(options)
            else:
                selected[dim] = options[0]
    elif random_sample:
        # Random sampling: each dimension independently
        for dim, options in STYLE_DIMENSIONS.items():
            selected[dim] = rand_module.choice(options)
    else:
        # Default: first option for each dimension
        for dim, options in STYLE_DIMENSIONS.items():
            selected[dim] = options[0]

    parts = [selected[dim] for dim in STYLE_DIMENSIONS.keys() if dim in selected]
    return " ".join(parts)


def get_all_style_combinations(max_combinations: int = 100) -> list:
    """Return all possible style combinations (or a random subset).

    Returns:
        List of style instruction strings.
    """
    import itertools
    import random

    all_options = [STYLE_DIMENSIONS[dim] for dim in STYLE_DIMENSIONS]
    all_combos = list(itertools.product(*all_options))

    if len(all_combos) > max_combinations:
        all_combos = random.sample(all_combos, max_combinations)

    results = []
    dims = list(STYLE_DIMENSIONS.keys())
    for combo in all_combos:
        parts = []
        for i, dim in enumerate(dims):
            parts.append(f"{dim.upper()}: {combo[i]}")
        results.append("\n".join(parts))

    return results


# =============================================================================
# Pre-generated style combinations
# =============================================================================

PRESET_STYLE_NAMES = list(QUALITY_PRESETS.keys())

STYLE_COMBINATIONS = [
    generate_style_instruction(quality=name)
    for name in PRESET_STYLE_NAMES
]


def get_dataset_optimal_style(dataset: str) -> str:
    """Return the recommended style instruction for the given dataset."""
    dataset_lower = (dataset or "").lower()

    if dataset_lower in ["drop", "hotpotqa"]:
        return generate_style_instruction(quality="CONCISE_DIRECT")
    elif dataset_lower in ["gsm8k", "math"]:
        return generate_style_instruction(quality="DETAILED_REASONING")
    elif dataset_lower in ["humaneval", "mbpp"]:
        return generate_style_instruction(quality="CODE_RIGOROUS")
    else:
        return generate_style_instruction(quality="SYSTEMATIC")


def get_random_style() -> str:
    """Generate a random style instruction (each dimension sampled independently)."""
    return generate_style_instruction(random_sample=True)


# =============================================================================
# GSM8K Math Prompts
# =============================================================================

GSM8K_SOLVE_PROMPT = """
Solve this math problem step by step. Show all your work clearly and end with a numerical answer.
Break down the solution into:
1. Given information
2. Step-by-step calculations
3. Final numerical answer clearly marked with ** **

Make sure to:
- Include all mathematical operations
- Show intermediate calculations
- Double check your arithmetic
- Consider all given values in the problem
- Verify your answer makes logical sense

Problem:
{input}
"""

GSM8K_EXTRACT_PROMPT = """
Extract only the final numerical answer from the solution. Return ONLY the number, with no text or symbols.
If there are multiple numbers, extract the one marked with ** **.
Compare this answer with the Programmer solution provided and return the Programmer's solution if it differs significantly.

Solution to extract from:
{input}
"""

# =============================================================================
# MBPP / HumanEval Code Generation Prompts
# =============================================================================

# Note: HumanEval's problem field already contains the full function signature and docstring.
# Uses {problem} and {entry_point} placeholders consistent with cached prompts.
CODE_GENERATE_PROMPT = """Generate a Python function to solve the given problem. Ensure the function name
matches the one specified in the problem. Include necessary imports. Use clear
variable names and add comments for clarity.

Problem:
{problem}

Function signature:
{entry_point}

Generate the complete function below:
"""

# CustomCodeGenerate prompt for HumanEval: designed for multi-solution ensemble.
# MASPO injects the full prompt, so it must be self-contained.
CUSTOM_CODE_GENERATE_PROMPT = """Generate clean, correct Python code to solve the following problem.

{problem}

Function name: {entry_point}

Generate the complete Python function only. No explanations or formatting.
"""

FIX_CODE_PROMPT = """
The provided solution failed to pass the tests. Please analyze the error and fix the
code. Ensure the function name and signature remain unchanged. If necessary, add
or modify imports, correct logical errors, and improve the implementation.

Problem:
{problem}

Failed Solution:
{solution}

Error Message:
{error}

Provide the corrected function below:
"""

# =============================================================================
# HumanEval Code Validation Prompts
# =============================================================================

VALIDATE_CODE_PROMPT = """
Analyze the given Python code and verify:
1. The function is properly defined with the specified name: {function_name}
2. The code contains a return statement
3. The code has valid Python syntax
4. The code is complete (no missing parts)

Code to validate:
{code}

Return only "VALID" if all checks pass, or "INVALID" if any check fails.
Do not include any other text in the response.
"""
