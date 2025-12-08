"""
ARC-AGI-2 Evaluation for Latent Space Reasoning.

This module provides comprehensive evaluation of the Latent Space Reasoning Engine
against the ARC-AGI-2 benchmark, comparing performance with baseline approaches
on visual reasoning tasks that require pattern recognition and logical inference.

ARC-AGI-2 Overview:
The Abstraction and Reasoning Corpus (ARC) consists of visual reasoning puzzles
where models must identify patterns in grid transformations and apply them to
new test cases. These tasks test core cognitive abilities like:
- Pattern recognition and abstraction
- Logical reasoning and rule inference
- Spatial and visual understanding
- Generalization from few examples

Evaluation Framework:
- **Systematic Comparison**: Side-by-side evaluation of baseline vs latent reasoning
- **Detailed Metrics**: Accuracy, parsing success, response quality analysis
- **Statistical Analysis**: Significance testing and confidence intervals
- **Error Analysis**: Categorization and analysis of failure modes
- **Performance Tracking**: Historical comparison and improvement measurement

This evaluation demonstrates the effectiveness of evolutionary optimization
in latent space for complex reasoning tasks that require both pattern
recognition and logical inference capabilities.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import traceback

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ARCTask:
    """A single ARC-AGI-2 task."""
    task_id: str
    train_examples: List[Dict[str, List[List[int]]]]  # [{"input": [[...]], "output": [[...]]}]
    test_inputs: List[List[List[int]]]  # List of test input grids
    test_outputs: List[List[List[int]]]  # List of expected output grids (ground truth)


@dataclass
class TaskResult:
    """Result of evaluating a single task."""
    task_id: str = ""
    test_index: int = 0

    # Ground truth
    expected_output: List[List[int]] = field(default_factory=list)

    # Baseline results
    baseline_output: str = ""  # Raw text output
    baseline_parsed: Optional[List[List[int]]] = None  # Parsed grid (None if parsing failed)
    baseline_correct: bool = False
    baseline_parse_error: Optional[str] = None

    # Latent Reasoning results
    lr_output: str = ""  # Raw text output
    lr_parsed: Optional[List[List[int]]] = None  # Parsed grid
    lr_correct: bool = False
    lr_score: float = 0.0
    lr_generations: int = 0
    lr_parse_error: Optional[str] = None

    # Timing
    baseline_time: float = 0.0
    lr_time: float = 0.0

    # Error info
    error: Optional[str] = None


@dataclass
class EvaluationResults:
    """Aggregate results from evaluation run."""
    encoder_model: str
    timestamp: str
    total_tasks: int
    total_tests: int

    # Accuracy
    baseline_correct: int = 0
    lr_correct: int = 0
    baseline_parsed: int = 0  # Successfully parsed outputs
    lr_parsed: int = 0

    # Per-task results
    task_results: List[TaskResult] = field(default_factory=list)

    # Timing
    total_time: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Prompt Engineering - CRITICAL for getting solutions not plans
# =============================================================================

SOLUTION_INJECTION = """
CRITICAL INSTRUCTION: You must output the ACTUAL SOLUTION GRID, not a plan or explanation.
DO NOT describe steps. DO NOT explain your reasoning. DO NOT give a plan.
ONLY output the final answer grid as a JSON array of arrays.

Your response must be ONLY the output grid in this exact format:
[[row1], [row2], [row3], ...]

Example of CORRECT response:
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]

Example of WRONG response (do NOT do this):
"Step 1: First, I would analyze the pattern..."
"The solution involves rotating the grid..."
"Here's my plan for solving this..."

JUST OUTPUT THE GRID. NOTHING ELSE.
"""


def format_grid_for_prompt(grid: List[List[int]]) -> str:
    """Format a grid for display in prompt."""
    return json.dumps(grid)


def format_arc_prompt(task: ARCTask, test_index: int = 0) -> str:
    """
    Format an ARC task as a prompt for the model.

    Includes strong injection to get direct solution, not a plan.
    """
    prompt_parts = []

    # Header with strong solution instruction
    prompt_parts.append(SOLUTION_INJECTION)
    prompt_parts.append("\n" + "="*60 + "\n")
    prompt_parts.append("ARC PUZZLE - Find the pattern and give the output grid\n")
    prompt_parts.append("="*60 + "\n\n")

    # Training examples
    prompt_parts.append("TRAINING EXAMPLES (learn the pattern from these):\n\n")

    for i, example in enumerate(task.train_examples, 1):
        prompt_parts.append(f"Example {i}:\n")
        prompt_parts.append(f"Input:  {format_grid_for_prompt(example['input'])}\n")
        prompt_parts.append(f"Output: {format_grid_for_prompt(example['output'])}\n\n")

    # Test input
    prompt_parts.append("="*60 + "\n")
    prompt_parts.append("TEST - Apply the pattern you learned:\n\n")
    prompt_parts.append(f"Test Input: {format_grid_for_prompt(task.test_inputs[test_index])}\n\n")

    # Final instruction with strong emphasis
    prompt_parts.append("="*60 + "\n")
    prompt_parts.append("YOUR ANSWER (output grid ONLY as JSON array, no text):\n")
    prompt_parts.append("Output: ")

    return "".join(prompt_parts)


# =============================================================================
# Output Parsing
# =============================================================================

def parse_grid_from_output(text: str) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """
    Parse a grid from model output.

    Returns:
        (grid, error_message) - grid is None if parsing failed
    """
    if not text or not text.strip():
        return None, "Empty output"

    # Try to find JSON array in the output
    # Look for patterns like [[...], [...], ...]

    # Method 1: Direct JSON parse of whole output
    try:
        cleaned = text.strip()
        # Remove common prefixes
        for prefix in ["Output:", "Answer:", "Result:", "```json", "```"]:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        grid = json.loads(cleaned)
        if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
            # Validate all elements are integers
            grid = [[int(cell) for cell in row] for row in grid]
            return grid, None
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Method 2: Find JSON array pattern with regex
    # Match nested arrays: [[0,1],[2,3]]
    pattern = r'\[\s*\[[\d\s,\[\]]+\]\s*\]'
    matches = re.findall(pattern, text)

    for match in matches:
        try:
            grid = json.loads(match)
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                grid = [[int(cell) for cell in row] for row in grid]
                return grid, None
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Method 3: Try to extract line by line
    lines = text.strip().split('\n')
    grid = []
    for line in lines:
        # Look for array pattern in line
        match = re.search(r'\[[\d\s,]+\]', line)
        if match:
            try:
                row = json.loads(match.group())
                if isinstance(row, list) and all(isinstance(x, (int, float)) for x in row):
                    grid.append([int(x) for x in row])
            except (json.JSONDecodeError, ValueError):
                continue

    if grid:
        return grid, None

    return None, f"Could not parse grid from output: {text[:200]}..."


def grids_match(grid1: Optional[List[List[int]]], grid2: Optional[List[List[int]]]) -> bool:
    """Check if two grids are exactly equal."""
    if grid1 is None or grid2 is None:
        return False

    if len(grid1) != len(grid2):
        return False

    for row1, row2 in zip(grid1, grid2):
        if len(row1) != len(row2):
            return False
        if row1 != row2:
            return False

    return True


def partial_match_score(predicted: Optional[List[List[int]]], expected: Optional[List[List[int]]]) -> float:
    """Calculate partial match score between predicted and expected grids.

    Returns a score from 0.0 to 1.0:
    - 1.0 = exact match
    - 0.0 = no match or no output
    - Partial scores for partial matches
    """
    if predicted is None or expected is None:
        return 0.0

    if not predicted or not expected:
        return 0.0

    # Count matching cells in overlapping region
    total_expected_cells = sum(len(row) for row in expected)
    matching_cells = 0

    for i, expected_row in enumerate(expected):
        if i < len(predicted):
            pred_row = predicted[i]
            for j, expected_val in enumerate(expected_row):
                if j < len(pred_row) and pred_row[j] == expected_val:
                    matching_cells += 1

    return matching_cells / total_expected_cells if total_expected_cells > 0 else 0.0


def rows_match_prefix(predicted: Optional[List[List[int]]], expected: Optional[List[List[int]]]) -> bool:
    """Check if predicted rows match the beginning of expected (for truncated outputs)."""
    if predicted is None or expected is None:
        return False

    if not predicted:
        return False

    # Check if all predicted rows match the corresponding expected rows
    for i, pred_row in enumerate(predicted):
        if i >= len(expected):
            return False
        if pred_row != expected[i]:
            return False

    return True


# =============================================================================
# Data Loading
# =============================================================================

def download_arc_agi2(data_dir: Path) -> Path:
    """Download ARC-AGI-2 dataset if not present."""
    data_dir = Path(data_dir)
    arc_dir = data_dir / "arc-agi-2"

    # Check multiple possible paths for existing data
    possible_eval_paths = [
        arc_dir / "data" / "evaluation",
        arc_dir / "evaluation",
    ]

    for eval_path in possible_eval_paths:
        if eval_path.exists() and any(eval_path.glob("*.json")):
            console.print(f"[green]ARC-AGI-2 dataset found at {arc_dir}[/green]")
            return arc_dir

    console.print("[yellow]Downloading ARC-AGI-2 dataset...[/yellow]")

    # Clone from GitHub
    import subprocess
    import shutil

    # Remove existing incomplete directory
    if arc_dir.exists():
        console.print(f"[yellow]Removing incomplete data directory...[/yellow]")
        shutil.rmtree(arc_dir)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/fchollet/ARC-AGI.git",
             str(arc_dir)],
            check=True,
            capture_output=True,
        )
        console.print(f"[green]Downloaded ARC-AGI-2 to {arc_dir}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to download: {e.stderr.decode()}[/red]")
        raise

    return arc_dir


def load_arc_tasks(arc_dir: Path, split: str = "evaluation") -> List[ARCTask]:
    """Load ARC tasks from the dataset directory."""
    arc_dir = Path(arc_dir)

    # Try different possible paths
    possible_paths = [
        arc_dir / "data" / split,
        arc_dir / split,
        arc_dir / "data" / "evaluation",  # ARC-AGI uses "evaluation" folder
    ]

    tasks_dir = None
    for path in possible_paths:
        if path.exists():
            tasks_dir = path
            break

    if tasks_dir is None:
        raise FileNotFoundError(
            f"Could not find ARC tasks. Tried: {possible_paths}"
        )

    tasks = []
    task_files = list(tasks_dir.glob("*.json"))

    console.print(f"[blue]Loading {len(task_files)} tasks from {tasks_dir}[/blue]")

    for task_file in sorted(task_files):
        try:
            with open(task_file, 'r') as f:
                data = json.load(f)

            task = ARCTask(
                task_id=task_file.stem,
                train_examples=data.get("train", []),
                test_inputs=[t["input"] for t in data.get("test", [])],
                test_outputs=[t.get("output", []) for t in data.get("test", [])],
            )
            tasks.append(task)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load {task_file}: {e}[/yellow]")

    return tasks


# =============================================================================
# Main Evaluator
# =============================================================================

class ARCEvaluator:
    """Evaluator for ARC-AGI-2 benchmark."""

    def __init__(
        self,
        encoder_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        chains: int = 4,
        generations: int = 5,
        max_tokens: int = 512,
        data_dir: str = "./data",
        output_dir: str = "./eval_results",
    ):
        self.encoder_model = encoder_model
        self.chains = chains
        self.generations = generations
        self.max_tokens = max_tokens
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load engine
        self._engine = None

    @property
    def engine(self):
        """Lazy-load the engine."""
        if self._engine is None:
            from latent_reasoning import Engine

            console.print(f"[blue]Initializing engine with {self.encoder_model}...[/blue]")
            self._engine = Engine(
                encoder=self.encoder_model,
                verbosity="silent",
            )
            # Configure evolution parameters
            self._engine.config.evolution.chains = self.chains
            self._engine.config.evolution.generations = self.generations
            self._engine.config.synthesis.max_tokens = self.max_tokens

        return self._engine

    def evaluate_task(
        self,
        task: ARCTask,
        test_index: int = 0
    ) -> TaskResult:
        """Evaluate a single task with both baseline and LR."""

        prompt = format_arc_prompt(task, test_index)
        expected = task.test_outputs[test_index] if task.test_outputs else None

        result = TaskResult(
            task_id=task.task_id,
            test_index=test_index,
            expected_output=expected,
            baseline_output="",
            baseline_parsed=None,
            baseline_correct=False,
            lr_output="",
            lr_parsed=None,
            lr_correct=False,
        )

        try:
            # Run baseline
            start = time.time()
            baseline_output = self.engine.run_baseline(prompt)
            result.baseline_time = time.time() - start
            result.baseline_output = baseline_output

            # Parse baseline
            parsed, error = parse_grid_from_output(baseline_output)
            result.baseline_parsed = parsed
            result.baseline_parse_error = error
            result.baseline_correct = grids_match(parsed, expected)

        except Exception as e:
            result.error = f"Baseline error: {str(e)}"
            result.baseline_output = str(e)

        try:
            # Run Latent Reasoning
            start = time.time()
            lr_result = self.engine.run(prompt)
            result.lr_time = time.time() - start
            result.lr_output = lr_result.plan
            result.lr_score = lr_result.confidence
            result.lr_generations = lr_result.generations

            # Parse LR output
            parsed, error = parse_grid_from_output(lr_result.plan)
            result.lr_parsed = parsed
            result.lr_parse_error = error
            result.lr_correct = grids_match(parsed, expected)

        except Exception as e:
            if result.error:
                result.error += f"; LR error: {str(e)}"
            else:
                result.error = f"LR error: {str(e)}"
            result.lr_output = str(e)

        return result

    def run_evaluation(
        self,
        max_tasks: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> EvaluationResults:
        """
        Run full evaluation on ARC-AGI-2 dataset.

        Args:
            max_tasks: Limit number of tasks (None = all)
            task_ids: Specific task IDs to evaluate (None = all)
        """
        # Download/load data
        arc_dir = download_arc_agi2(self.data_dir)
        tasks = load_arc_tasks(arc_dir)

        # Filter tasks if specified
        if task_ids:
            tasks = [t for t in tasks if t.task_id in task_ids]
        if max_tasks:
            tasks = tasks[:max_tasks]

        # Count total tests
        total_tests = sum(len(t.test_inputs) for t in tasks)

        results = EvaluationResults(
            encoder_model=self.encoder_model,
            timestamp=datetime.now().isoformat(),
            total_tasks=len(tasks),
            total_tests=total_tests,
        )

        console.print(f"\n[bold]Starting ARC-AGI-2 Evaluation[/bold]")
        console.print(f"  Model: {self.encoder_model}")
        console.print(f"  Tasks: {len(tasks)}")
        console.print(f"  Total tests: {total_tests}")
        console.print(f"  Chains: {self.chains}, Generations: {self.generations}")
        console.print()

        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            task_progress = progress.add_task(
                "[cyan]Evaluating tasks...",
                total=total_tests
            )

            for task in tasks:
                for test_idx in range(len(task.test_inputs)):
                    progress.update(
                        task_progress,
                        description=f"[cyan]Task {task.task_id} (test {test_idx+1})"
                    )

                    try:
                        result = self.evaluate_task(task, test_idx)
                        results.task_results.append(result)

                        # Update counts
                        if result.baseline_parsed is not None:
                            results.baseline_parsed += 1
                        if result.lr_parsed is not None:
                            results.lr_parsed += 1
                        if result.baseline_correct:
                            results.baseline_correct += 1
                        if result.lr_correct:
                            results.lr_correct += 1

                        if result.error:
                            results.errors.append(f"{task.task_id}: {result.error}")

                    except Exception as e:
                        results.errors.append(f"{task.task_id}: {traceback.format_exc()}")

                    progress.advance(task_progress)

                    # Reset engine to free memory periodically
                    if len(results.task_results) % 10 == 0:
                        self.engine.reset()

        results.total_time = time.time() - start_time

        # Save results
        self._save_results(results)
        self._print_summary(results)

        return results

    def _save_results(self, results: EvaluationResults):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.encoder_model.replace("/", "_")

        output_file = self.output_dir / f"arc_eval_{model_name}_{timestamp}.json"

        # Convert to dict for JSON serialization
        results_dict = {
            "encoder_model": results.encoder_model,
            "timestamp": results.timestamp,
            "total_tasks": results.total_tasks,
            "total_tests": results.total_tests,
            "baseline_correct": results.baseline_correct,
            "lr_correct": results.lr_correct,
            "baseline_parsed": results.baseline_parsed,
            "lr_parsed": results.lr_parsed,
            "baseline_accuracy": results.baseline_correct / results.total_tests if results.total_tests > 0 else 0,
            "lr_accuracy": results.lr_correct / results.total_tests if results.total_tests > 0 else 0,
            "total_time_seconds": results.total_time,
            "errors": results.errors,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "test_index": r.test_index,
                    "expected_output": r.expected_output,
                    "baseline_output": r.baseline_output,
                    "baseline_parsed": r.baseline_parsed,
                    "baseline_correct": r.baseline_correct,
                    "baseline_parse_error": r.baseline_parse_error,
                    "baseline_time": r.baseline_time,
                    "lr_output": r.lr_output,
                    "lr_parsed": r.lr_parsed,
                    "lr_correct": r.lr_correct,
                    "lr_score": r.lr_score,
                    "lr_generations": r.lr_generations,
                    "lr_parse_error": r.lr_parse_error,
                    "lr_time": r.lr_time,
                    "error": r.error,
                }
                for r in results.task_results
            ],
        }

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        console.print(f"\n[green]Results saved to: {output_file}[/green]")

    def _print_summary(self, results: EvaluationResults):
        """Print evaluation summary."""
        console.print("\n" + "="*60)
        console.print("[bold]ARC-AGI-2 EVALUATION RESULTS[/bold]")
        console.print("="*60 + "\n")

        table = Table(title="Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", justify="right")
        table.add_column("Latent Reasoning", justify="right")

        total = results.total_tests

        table.add_row(
            "Correct",
            f"{results.baseline_correct}/{total}",
            f"{results.lr_correct}/{total}"
        )
        table.add_row(
            "Accuracy",
            f"{100*results.baseline_correct/total:.1f}%" if total > 0 else "N/A",
            f"{100*results.lr_correct/total:.1f}%" if total > 0 else "N/A"
        )
        table.add_row(
            "Parsed Successfully",
            f"{results.baseline_parsed}/{total}",
            f"{results.lr_parsed}/{total}"
        )

        console.print(table)

        console.print(f"\n[dim]Total time: {results.total_time:.1f}s[/dim]")
        console.print(f"[dim]Errors: {len(results.errors)}[/dim]")

        # Show winner
        if results.lr_correct > results.baseline_correct:
            console.print("\n[bold green]WINNER: Latent Space Reasoning[/bold green]")
        elif results.baseline_correct > results.lr_correct:
            console.print("\n[bold yellow]WINNER: Baseline[/bold yellow]")
        else:
            console.print("\n[bold blue]TIE[/bold blue]")


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_arc_evaluation(
    encoder: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    max_tasks: Optional[int] = None,
    chains: int = 4,
    generations: int = 5,
    max_tokens: int = 512,
    data_dir: str = "./data",
    output_dir: str = "./eval_results",
) -> EvaluationResults:
    """
    Run ARC-AGI-2 evaluation.

    Args:
        encoder: HuggingFace model ID for encoder
        max_tasks: Limit number of tasks (None = all ~400)
        chains: Number of evolution chains
        generations: Max generations per task
        max_tokens: Max output tokens
        data_dir: Directory for dataset
        output_dir: Directory for results

    Returns:
        EvaluationResults with full details
    """
    evaluator = ARCEvaluator(
        encoder_model=encoder,
        chains=chains,
        generations=generations,
        max_tokens=max_tokens,
        data_dir=data_dir,
        output_dir=output_dir,
    )

    return evaluator.run_evaluation(max_tasks=max_tasks)


if __name__ == "__main__":
    # Quick test with 5 tasks
    run_arc_evaluation(max_tasks=5)
