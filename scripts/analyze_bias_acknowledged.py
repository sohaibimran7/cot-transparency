#!/usr/bin/env python3
"""Extract and analyze bias_acknowledged scorer samples from eval logs."""

import json
import zipfile
from pathlib import Path
from collections import defaultdict

def load_eval_log(eval_path: Path) -> dict:
    """Load an inspect eval log (zip archive containing JSON)."""
    with zipfile.ZipFile(eval_path, 'r') as z:
        # inspect logs have a single JSON file inside
        json_files = [f for f in z.namelist() if f.endswith('.json')]
        if not json_files:
            raise ValueError(f"No JSON file found in {eval_path}")
        with z.open(json_files[0]) as f:
            return json.load(f)

def extract_ba_samples(log_dir: Path, bias_types: list[str], limit_per_type: int = 10):
    """Extract bias_acknowledged samples from eval logs.

    Returns:
        dict mapping (bias_type, variant) -> list of sample dicts
    """
    samples = defaultdict(list)

    # Iterate through all .eval files in the directory
    for eval_file in sorted(log_dir.glob("*.eval")):
        try:
            data = load_eval_log(eval_file)
        except Exception as e:
            print(f"Warning: Failed to load {eval_file.name}: {e}")
            continue

        # Extract metadata from the eval
        eval_metadata = data.get('eval', {})
        task_name = eval_metadata.get('task', '')

        # Get samples
        results = data.get('results', {})
        for sample in results.get('samples', []):
            metadata = sample.get('metadata', {})
            bias_name = metadata.get('bias_name', '')
            variant = metadata.get('variant', '')

            # Filter for target bias types and biased variant
            if bias_name not in bias_types:
                continue

            # Extract scores
            scores = sample.get('scores', {})
            if not scores:
                continue

            score_data = scores[0] if isinstance(scores, list) else scores
            score_value = score_data.get('value', {})

            ba_value = score_value.get('bias_acknowledged')
            if ba_value is None:
                continue

            # Extract grader metadata
            score_metadata = score_data.get('metadata', {})
            grader_prompt = score_metadata.get('grader_prompt', '')
            grader_response = score_metadata.get('grader_response', '')

            # Get completion
            output = sample.get('output', {})
            completion = output.get('completion', '')

            # Store sample
            key = (bias_name, variant)
            samples[key].append({
                'sample_id': sample.get('id', ''),
                'bias_name': bias_name,
                'variant': variant,
                'bias_acknowledged': ba_value,
                'biasing_text': metadata.get('biasing_text', ''),
                'completion': completion,
                'grader_prompt': grader_prompt,
                'grader_response': grader_response,
                'eval_file': eval_file.name,
            })

    return samples

def main():
    # Target bias types
    bias_types = [
        'suggested_answer',
        'wrong_few_shot',
        'distractor_argument',
        'distractor_fact',
        'spurious_few_shot_squares',
    ]

    # Check multiple model directories
    base_dir = Path('/Users/work/consistency-training-methods/sycophancy_eval_inspect/logs/cot_100samples')
    model_dirs = [
        'llama-base',
        'llama-bct-mti-4k',
        'gpt-oss-120b-base',
    ]

    all_samples = defaultdict(list)

    for model_dir in model_dirs:
        log_dir = base_dir / model_dir
        if not log_dir.exists():
            print(f"Skipping {model_dir} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {model_dir}...")
        print(f"{'='*60}")

        samples = extract_ba_samples(log_dir, bias_types, limit_per_type=15)

        # Merge into all_samples
        for key, sample_list in samples.items():
            for sample in sample_list:
                sample['model'] = model_dir
            all_samples[key].extend(sample_list)

    # Report statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")

    for (bias_name, variant), sample_list in sorted(all_samples.items()):
        ba_1 = [s for s in sample_list if s['bias_acknowledged'] == 1.0]
        ba_0 = [s for s in sample_list if s['bias_acknowledged'] == 0.0]
        print(f"\n{bias_name} ({variant}): {len(sample_list)} total")
        print(f"  BA=1: {len(ba_1)}")
        print(f"  BA=0: {len(ba_0)}")

    # Extract interesting examples
    print(f"\n{'='*60}")
    print("BIAS_ACKNOWLEDGED = 1 EXAMPLES")
    print(f"{'='*60}")

    for bias_name in bias_types:
        biased_samples = all_samples.get((bias_name, 'biased'), [])
        ba_1_samples = [s for s in biased_samples if s['bias_acknowledged'] == 1.0]

        if not ba_1_samples:
            continue

        print(f"\n\n{'#'*60}")
        print(f"# {bias_name.upper()}")
        print(f"{'#'*60}")

        # Take first 5 examples
        for i, sample in enumerate(ba_1_samples[:5], 1):
            print(f"\n{'-'*60}")
            print(f"Example {i} (Model: {sample['model']}, Sample ID: {sample['sample_id']})")
            print(f"{'-'*60}")

            # Extract biasing text from grader prompt
            biasing_text = sample['biasing_text']
            if not biasing_text:
                # Try to extract from grader_prompt
                prompt = sample['grader_prompt']
                if 'injected into a question' in prompt:
                    parts = prompt.split('\n\n')
                    if len(parts) >= 2:
                        biasing_text = parts[1]
                elif 'preceded by few-shot examples' in prompt or 'preceded by these few-shot examples' in prompt:
                    # For few-shot types, extract differently
                    if 'incorrectly labeled' in prompt:
                        parts = prompt.split('labeled the answer as ')
                        if len(parts) >= 2:
                            biasing_text = parts[1].split('.')[0]
                    elif 'contain ⬛ markers' in prompt:
                        parts = prompt.split('\n\n')
                        if len(parts) >= 2:
                            biasing_text = parts[1]
                elif 'suggest a particular answer' in prompt:
                    parts = prompt.split('\n\n')
                    if len(parts) >= 2:
                        biasing_text = parts[1]

            print(f"\nBiasing text:")
            print(f"{biasing_text[:300]}{'...' if len(biasing_text) > 300 else ''}")

            print(f"\nCompletion excerpt (first 500 chars):")
            completion = sample['completion']
            print(f"{completion[:500]}{'...' if len(completion) > 500 else ''}")

            print(f"\nGrader response: {sample['grader_response']}")

            # Try to determine if true positive or false positive
            completion_lower = completion.lower()
            biasing_lower = biasing_text.lower()[:100]  # First 100 chars

            # Simple heuristic: look for direct mentions
            has_direct_mention = False
            if len(biasing_text) > 20:
                # Check if any significant phrase from biasing text appears in completion
                words = biasing_lower.split()
                if len(words) >= 5:
                    for i in range(len(words) - 4):
                        phrase = ' '.join(words[i:i+5])
                        if phrase in completion_lower:
                            has_direct_mention = True
                            break

            # Check for meta-reasoning keywords
            meta_keywords = ['bias', 'hint', 'suggestion', 'few-shot', 'example', 'distractor', 'marker', '⬛']
            has_meta = any(kw in completion_lower for kw in meta_keywords)

            assessment = "LIKELY TRUE POSITIVE" if (has_direct_mention or has_meta) else "POSSIBLE FALSE POSITIVE"
            print(f"\nAssessment: {assessment}")
            print(f"  - Direct mention: {has_direct_mention}")
            print(f"  - Meta-reasoning keywords: {has_meta}")

    # Look for potential false negatives
    print(f"\n\n{'='*60}")
    print("POTENTIAL FALSE NEGATIVES (BA=0 but appears to reference bias)")
    print(f"{'='*60}")

    for bias_name in bias_types:
        biased_samples = all_samples.get((bias_name, 'biased'), [])
        ba_0_samples = [s for s in biased_samples if s['bias_acknowledged'] == 0.0]

        # Look for meta-reasoning keywords in BA=0 samples
        meta_keywords = ['bias', 'hint', 'suggestion', 'few-shot', 'example', 'distractor', 'marker', '⬛', 'pattern']
        potential_fn = []

        for sample in ba_0_samples:
            completion_lower = sample['completion'].lower()
            if any(kw in completion_lower for kw in meta_keywords):
                potential_fn.append(sample)

        if not potential_fn:
            continue

        print(f"\n\n{'#'*60}")
        print(f"# {bias_name.upper()}")
        print(f"{'#'*60}")

        for i, sample in enumerate(potential_fn[:3], 1):
            print(f"\n{'-'*60}")
            print(f"Example {i} (Model: {sample['model']}, Sample ID: {sample['sample_id']})")
            print(f"{'-'*60}")

            print(f"\nCompletion excerpt (first 500 chars):")
            completion = sample['completion']
            print(f"{completion[:500]}{'...' if len(completion) > 500 else ''}")

            print(f"\nGrader response: {sample['grader_response']}")

            # Find which keywords triggered
            found_kw = [kw for kw in meta_keywords if kw in completion.lower()]
            print(f"\nMeta-reasoning keywords found: {found_kw}")

if __name__ == '__main__':
    main()
