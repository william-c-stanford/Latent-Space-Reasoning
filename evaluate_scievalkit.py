#!/usr/bin/env python3
"""
SciEvalKit Evaluation Script for Latent Space Reasoning

This script evaluates Qwen/Qwen3-0.6B with and without latent space reasoning
on text-only scientific benchmarks from SciEvalKit.

Text-only datasets available:
- MaScQA: Materials Science Question Answering
- ProteinLMBench: Protein Language Model Benchmark
- PHYSICS: Physics problems
- TRQA: Text-based Reasoning QA
- AstroVisBench: Astronomy benchmarks (text component)

Usage:
    python evaluate_scievalkit.py --dataset MaScQA --mode all
    python evaluate_scievalkit.py --dataset ProteinLMBench --mode infer
"""

import argparse
import json
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / 'src'))
sys.path.insert(0, str(SCRIPT_DIR / 'SciEvalKit'))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def get_text_only_datasets():
    """Return list of text-only datasets suitable for LLM evaluation."""
    return [
        'MaScQA',           # Materials Science QA - good starter
        'ProteinLMBench',   # Protein LM benchmark
        # 'PHYSICS',        # Physics problems (may need special setup)
        # 'TRQA',           # Text reasoning QA
    ]


def setup_model_configs():
    """Set up model configurations for SciEvalKit."""
    from latent_reasoning.eval.scievalkit_wrapper import BaselineQwenModel, LatentReasoningModel
    
    configs = {
        'Qwen3-0.6B-Baseline': partial(
            BaselineQwenModel,
            model_path='Qwen/Qwen3-0.6B',
            quantization='4bit',
            max_new_tokens=1024,
            temperature=0.7,
        ),
        'Qwen3-0.6B-LatentReasoning': partial(
            LatentReasoningModel,
            model_path='Qwen/Qwen3-0.6B',
            quantization='4bit',
            max_new_tokens=1024,
            temperature=0.7,
            chains=5,
            generations=10,
            verbosity='minimal',
        ),
    }
    return configs


def run_evaluation(
    dataset_name: str,
    model_name: str,
    model_factory,
    work_dir: str,
    mode: str = 'all',
    verbose: bool = False,
):
    """
    Run evaluation for a single model on a single dataset.
    
    Args:
        dataset_name: Name of dataset to evaluate
        model_name: Name of model configuration
        model_factory: Callable that creates the model
        work_dir: Output directory
        mode: 'all' (infer+eval), 'infer' (inference only), 'eval' (eval only)
        verbose: Print verbose output
    """
    from scieval.dataset import build_dataset
    from scieval.smp import load, dump, get_pred_file_path
    from tqdm import tqdm
    import pandas as pd
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} on {dataset_name}")
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")
    
    # Create output directory
    model_work_dir = os.path.join(work_dir, model_name)
    os.makedirs(model_work_dir, exist_ok=True)
    
    # Build dataset - import specific dataset classes to handle different init signatures
    print(f"Loading dataset: {dataset_name}...")
    try:
        if dataset_name == 'MaScQA':
            from scieval.dataset import MaScQA
            dataset = MaScQA()
        elif dataset_name == 'ProteinLMBench':
            from scieval.dataset import ProteinLMBench
            dataset = ProteinLMBench(dataset='ProteinLMBench')
        else:
            dataset = build_dataset(dataset_name)
    except Exception as e:
        print(f"ERROR: Failed to load dataset {dataset_name}: {e}")
        return None
        
    if dataset is None:
        print(f"ERROR: Failed to load dataset {dataset_name}")
        return None
        
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Dataset type: {dataset.TYPE}, Modality: {dataset.MODALITY}")
    
    # Define result file path
    result_file = os.path.join(model_work_dir, f'{model_name}_{dataset_name}.pkl')
    
    # Run inference if needed
    if mode in ['all', 'infer']:
        print(f"\nStarting inference...")
        
        # Load existing results if any
        if os.path.exists(result_file):
            existing_data = load(result_file)
            existing_indices = set(existing_data['index']) if 'index' in existing_data else set()
            print(f"Found {len(existing_indices)} existing predictions")
        else:
            existing_indices = set()
            
        # Build model
        print(f"Loading model: {model_name}...")
        model = model_factory()
        model.set_dump_image(dataset.dump_image)
        
        # Run inference
        data = dataset.data
        results = {}
        
        # Load any partial results
        partial_file = os.path.join(model_work_dir, f'{model_name}_{dataset_name}_partial.pkl')
        if os.path.exists(partial_file):
            results = load(partial_file)
            print(f"Loaded {len(results)} partial results")
            
        samples_to_run = [i for i in range(len(data)) if data.iloc[i]['index'] not in results]
        print(f"Running inference on {len(samples_to_run)} samples...")
        
        for i in tqdm(samples_to_run, desc=f'{model_name}/{dataset_name}'):
            idx = data.iloc[i]['index']
            if idx in results:
                continue
                
            # Build prompt
            prompt = dataset.build_prompt(data.iloc[i])
            
            try:
                response = model.generate(message=prompt, dataset=dataset_name)
            except Exception as e:
                print(f"\nError on sample {idx}: {e}")
                response = f"ERROR: {str(e)}"
                
            results[idx] = response
            
            if verbose:
                print(f"\n--- Sample {idx} ---")
                print(f"Prompt: {prompt[0]['value'][:200]}...")
                print(f"Response: {response[:200]}...")
                
            # Save partial results every 10 samples
            if (i + 1) % 10 == 0:
                dump(results, partial_file)
                
        # Save final results
        data_copy = data.copy()
        data_copy['prediction'] = [str(results.get(x, '')) for x in data_copy['index']]
        dump(data_copy, result_file)
        
        # Clean up partial file
        if os.path.exists(partial_file):
            os.remove(partial_file)
            
        print(f"Inference complete. Results saved to: {result_file}")
        
    # Run evaluation if needed
    if mode in ['all', 'eval']:
        print(f"\nStarting evaluation...")
        
        if not os.path.exists(result_file):
            print(f"ERROR: No prediction file found at {result_file}")
            print("Run with --mode infer first")
            return None
            
        # Run dataset evaluation
        judge_kwargs = {
            'nproc': 4,
            'verbose': verbose,
            'work_dir': work_dir,
            'eval_model_name': model_name,
            'model': 'exact_matching',  # Use exact matching to avoid API costs
        }
        
        try:
            eval_results = dataset.evaluate(result_file, **judge_kwargs)
            print(f"\n{'='*40}")
            print(f"RESULTS: {model_name} on {dataset_name}")
            print(f"{'='*40}")
            if isinstance(eval_results, dict):
                print(json.dumps(eval_results, indent=2))
            else:
                print(eval_results)
            return eval_results
        except Exception as e:
            print(f"Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    return None


def compare_results(results: dict, output_file: str = None):
    """Compare results across models and generate report."""
    import pandas as pd
    
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60 + "\n")
    
    # Organize by dataset
    by_dataset = {}
    for key, value in results.items():
        model, dataset = key.rsplit('_', 1)
        if dataset not in by_dataset:
            by_dataset[dataset] = {}
        by_dataset[dataset][model] = value
        
    report_lines = []
    report_lines.append("# SciEvalKit Evaluation: Qwen3-0.6B with Latent Space Reasoning\n")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for dataset, model_results in by_dataset.items():
        report_lines.append(f"\n## {dataset}\n")
        print(f"\n{dataset}:")
        print("-" * 40)
        
        for model, result in model_results.items():
            if result is not None:
                if isinstance(result, pd.DataFrame):
                    result_str = result.to_string()
                elif isinstance(result, dict):
                    result_str = json.dumps(result, indent=2)
                else:
                    result_str = str(result)
                    
                print(f"\n{model}:")
                print(result_str)
                report_lines.append(f"\n### {model}\n")
                report_lines.append(f"```\n{result_str}\n```\n")
                
    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Qwen3-0.6B with and without Latent Space Reasoning on SciEvalKit'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        nargs='+',
        default=['MaScQA'],
        help='Dataset(s) to evaluate. Options: MaScQA, ProteinLMBench'
    )
    parser.add_argument(
        '--model',
        type=str,
        nargs='+',
        default=['Qwen3-0.6B-Baseline', 'Qwen3-0.6B-LatentReasoning'],
        help='Model(s) to evaluate'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'infer', 'eval'],
        help='Mode: all (infer+eval), infer (inference only), eval (evaluation only)'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./outputs/scievalkit_comparison',
        help='Output directory'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit number of samples (for testing)'
    )
    
    args = parser.parse_args()
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Get model configs
    model_configs = setup_model_configs()
    
    # Filter models
    selected_models = {k: v for k, v in model_configs.items() if k in args.model}
    if not selected_models:
        print(f"ERROR: No valid models selected. Available: {list(model_configs.keys())}")
        return
        
    print(f"Models to evaluate: {list(selected_models.keys())}")
    print(f"Datasets to evaluate: {args.dataset}")
    print(f"Work directory: {args.work_dir}")
    
    # Run evaluations
    all_results = {}
    
    for dataset_name in args.dataset:
        for model_name, model_factory in selected_models.items():
            result = run_evaluation(
                dataset_name=dataset_name,
                model_name=model_name,
                model_factory=model_factory,
                work_dir=args.work_dir,
                mode=args.mode,
                verbose=args.verbose,
            )
            all_results[f'{model_name}_{dataset_name}'] = result
            
    # Generate comparison report
    if args.mode in ['all', 'eval']:
        report_file = os.path.join(args.work_dir, 'comparison_report.md')
        compare_results(all_results, report_file)
        
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

