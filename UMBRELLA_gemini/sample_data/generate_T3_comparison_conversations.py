#!/usr/bin/env python3
"""
Generalized T3 Comparison Conversation Generator
=================================================

Generates multi-turn T3 comparison conversations from split/pair CSVs
produced by create_T3_comparison_dataset.py.

This replaces the sex-specific conversation generators with a fully
parameterized version that works with any target variable.

Conversation Pattern (T3):
    Turn 1 (User):      [reference image] + classification prompt
    Turn 1 (Assistant):  reference label
    Turn 2 (User):      [target image] + comparison prompt
    Turn 2 (Assistant):  target label

Usage Example:
    python generate_T3_comparison_conversations.py \
        --splits-dir my_comparison_splits \
        --output-dir my_comparison_conversations \
        --image-dir /path/to/images \
        --image-pattern "{subject_id}.nii.gz" \
        --target-name "biological sex" \
        --modality sMRI

Output:
    <output-dir>/
        train_conversations.jsonl
        validation_conversations.jsonl
        test_conversations.jsonl
        config.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Default prompt templates
# =============================================================================
# {target_name} and {modality_desc} are replaced at runtime.

DEFAULT_PROMPTS = {
    'reference': (
        "Analyze this {modality_desc} brain MRI scan (reference). "
        "Estimate the {target_name} of the subject."
    ),
    'comparison': (
        "Now analyze this {modality_desc} MRI by comparing it with the "
        "reference scan. Estimate the {target_name} of this subject."
    ),
}

MODALITY_DESC = {
    'sMRI': 'T1-weighted structural',
    'fMRI': 'functional',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate T3 comparison conversations from split/pair data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_T3_comparison_conversations.py \\
      --splits-dir splits_sex_100 \\
      --output-dir conversations_sex_100 \\
      --image-dir /data/ABCD/sMRI \\
      --target-name "biological sex" \\
      --modality sMRI

  # With custom image pattern and fMRI
  python generate_T3_comparison_conversations.py \\
      --splits-dir splits_diagnosis \\
      --output-dir conversations_diagnosis \\
      --image-dir /data/fMRI \\
      --image-pattern "sub-{subject_id}_bold.nii.gz" \\
      --target-name "clinical diagnosis" \\
      --modality fMRI
        """
    )

    # --- Required ---
    parser.add_argument("--splits-dir", type=str, required=True,
                        help="Directory with split/pair CSVs "
                             "(output of create_T3_comparison_dataset.py)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for conversation JSONL files")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing brain image files")

    # --- Optional ---
    parser.add_argument("--image-pattern", type=str,
                        default="{subject_id}.nii.gz",
                        help="Image filename pattern with {subject_id} placeholder "
                             "(default: {subject_id}.nii.gz)")
    parser.add_argument("--target-name", type=str,
                        default="target attribute",
                        help="Human-readable target name for prompts "
                             "(e.g. 'biological sex', 'age group')")
    parser.add_argument("--modality", type=str, default="sMRI",
                        choices=["sMRI", "fMRI"],
                        help="Imaging modality (default: sMRI)")
    parser.add_argument("--splits", nargs='+',
                        default=['train', 'validation', 'test'],
                        help="Which splits to generate (default: all)")
    parser.add_argument("--reference-prompt", type=str, default=None,
                        help="Custom reference prompt template "
                             "(can use {target_name}, {modality_desc})")
    parser.add_argument("--comparison-prompt", type=str, default=None,
                        help="Custom comparison prompt template "
                             "(can use {target_name}, {modality_desc})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser.parse_args()


# =============================================================================
# Conversation Builder
# =============================================================================

def get_image_path(image_dir: str, image_pattern: str, subject_id: str) -> str:
    """Build full image path for a subject."""
    filename = image_pattern.format(subject_id=subject_id)
    return f"{image_dir}/{filename}"


def create_conversation(
    subject_id: str,
    subject_target: str,
    reference_id: str,
    reference_target: str,
    comparison_type: str,
    image_dir: str,
    image_pattern: str,
    ref_prompt: str,
    comp_prompt: str,
    modality: str,
) -> Dict:
    """
    Create a single T3 multi-turn comparison conversation.

    Structure:
        Turn 1: User shows reference image + asks classification
                 Assistant answers with reference label
        Turn 2: User shows target image + asks comparison-based classification
                 Assistant answers with target label

    Args:
        subject_id: Target subject ID
        subject_target: Target subject's label
        reference_id: Reference subject ID
        reference_target: Reference subject's label
        comparison_type: 'same' or 'different'
        image_dir: Image directory path
        image_pattern: Filename pattern with {subject_id}
        ref_prompt: Formatted reference prompt
        comp_prompt: Formatted comparison prompt
        modality: Imaging modality (sMRI/fMRI)

    Returns:
        Conversation dictionary in LLaVA-compatible format
    """
    ref_image = get_image_path(image_dir, image_pattern, reference_id)
    subj_image = get_image_path(image_dir, image_pattern, subject_id)

    conversation = {
        "task_id": f"{subject_id}_{comparison_type}_comparison",
        "task_type": "T3",
        "subject_ids": [reference_id, subject_id],
        "modalities": [modality, modality],
        "images": [
            {"path": ref_image, "token": "<image>", "modality": modality},
            {"path": subj_image, "token": "<image>", "modality": modality},
        ],
        "conversations": [
            # Turn 1: Reference classification
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ref_prompt},
                    {"type": "image", "modality": modality, "image_path": ref_image},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{reference_target}."}
                ]
            },
            # Turn 2: Comparison-based classification
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": comp_prompt},
                    {"type": "image", "modality": modality, "image_path": subj_image},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{subject_target}."}
                ]
            },
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": subject_target,
            "reference_id": reference_id,
            "reference_label": reference_target,
            "comparison_type": comparison_type,
            "task": "comparison",
        }
    }

    return conversation


# =============================================================================
# Generation Logic
# =============================================================================

def generate_conversations_for_split(
    split_name: str,
    pairs_df: pd.DataFrame,
    image_dir: str,
    image_pattern: str,
    ref_prompt: str,
    comp_prompt: str,
    modality: str,
) -> List[Dict]:
    """
    Generate all T3 conversations for a given split.

    Args:
        split_name: Split name (train/validation/test)
        pairs_df: DataFrame with pair metadata
        image_dir: Image directory
        image_pattern: Image filename pattern
        ref_prompt: Formatted reference prompt
        comp_prompt: Formatted comparison prompt
        modality: Imaging modality

    Returns:
        List of conversation dictionaries
    """
    conversations = []
    errors = 0

    for _, row in pairs_df.iterrows():
        try:
            conv = create_conversation(
                subject_id=str(row['subject_id']),
                subject_target=str(row['target']),
                reference_id=str(row['reference_id']),
                reference_target=str(row['reference_target']),
                comparison_type=str(row['comparison_type']),
                image_dir=image_dir,
                image_pattern=image_pattern,
                ref_prompt=ref_prompt,
                comp_prompt=comp_prompt,
                modality=modality,
            )
            conversations.append(conv)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning: {row['subject_id']}: {e}")

    if errors > 5:
        print(f"  ... and {errors - 5} more errors")

    return conversations


def save_conversations(
    split_name: str,
    conversations: List[Dict],
    output_dir: Path,
):
    """Save conversations to JSONL file."""
    jsonl_path = output_dir / f"{split_name}_conversations.jsonl"

    with open(jsonl_path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

    print(f"  {split_name}: {len(conversations)} conversations -> {jsonl_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("T3 COMPARISON CONVERSATION GENERATOR (Generalized)")
    print("=" * 60)
    print(f"Splits dir: {splits_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Image dir: {args.image_dir}")
    print(f"Image pattern: {args.image_pattern}")
    print(f"Target name: {args.target_name}")
    print(f"Modality: {args.modality}")

    if not splits_dir.exists():
        print(f"\nError: Splits directory not found: {splits_dir}")
        print("Run create_T3_comparison_dataset.py first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt templates
    modality_desc = MODALITY_DESC.get(args.modality, args.modality)

    if args.reference_prompt:
        ref_prompt = args.reference_prompt.format(
            target_name=args.target_name, modality_desc=modality_desc
        )
    else:
        ref_prompt = DEFAULT_PROMPTS['reference'].format(
            target_name=args.target_name, modality_desc=modality_desc
        )

    if args.comparison_prompt:
        comp_prompt = args.comparison_prompt.format(
            target_name=args.target_name, modality_desc=modality_desc
        )
    else:
        comp_prompt = DEFAULT_PROMPTS['comparison'].format(
            target_name=args.target_name, modality_desc=modality_desc
        )

    print(f"\nPrompts:")
    print(f"  Reference:  {ref_prompt}")
    print(f"  Comparison: {comp_prompt}")

    # Process each split
    for split_name in args.splits:
        pairs_path = splits_dir / f"{split_name}_pairs.csv"

        if not pairs_path.exists():
            print(f"\nWarning: {pairs_path} not found, skipping {split_name}.")
            continue

        print(f"\nProcessing {split_name.upper()}:")
        pairs_df = pd.read_csv(pairs_path)
        print(f"  Loaded {len(pairs_df)} pairs")

        conversations = generate_conversations_for_split(
            split_name=split_name,
            pairs_df=pairs_df,
            image_dir=args.image_dir,
            image_pattern=args.image_pattern,
            ref_prompt=ref_prompt,
            comp_prompt=comp_prompt,
            modality=args.modality,
        )

        save_conversations(split_name, conversations, output_dir)

    # Save config for reproducibility
    config = vars(args)
    config['ref_prompt_resolved'] = ref_prompt
    config['comp_prompt_resolved'] = comp_prompt
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Output: {output_dir.absolute()}")
    print(f"\nNext step (training):")
    print(f"  python project/training/main_umbrella_training.py \\")
    print(f"      --config project/config/umbrella_llava_train.yaml \\")
    print(f"      --train-data {output_dir}/train_conversations.jsonl \\")
    print(f"      --eval-data {output_dir}/validation_conversations.jsonl \\")
    print(f"      --modality {args.modality} \\")
    print(f"      --output-dir ./hf_results/<experiment_name> \\")
    print(f"      --eval-output-dir ./eval_predictions_<experiment_name>")


if __name__ == "__main__":
    main()
