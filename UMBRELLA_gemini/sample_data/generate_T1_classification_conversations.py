#!/usr/bin/env python3
"""
T1 Direct Prediction Conversation Generator
============================================

Creates single-turn T1 conversations for DIRECT PREDICTION (no comparison).

Ablation Purpose (Experiment 1):
    Tests whether model performance is due to the comparison structure (T3)
    or whether direct prediction alone (T1) is equally effective.
    Expected: T1 should show resolution-dependent performance differences.

Conversation Pattern (T1):
    Turn 1 (User):      [target image] + classification prompt
    Turn 1 (Assistant):  label

vs. T3 Pattern:
    Turn 1: [ref image] → ref label
    Turn 2: [target image] → target label (with comparison context)

Usage:
    python generate_T1_classification_conversations.py \
        --splits-dir HCvsMCI_splits_300subj \
        --output-dir HCvsMCI_T1_conversations_300subj \
        --image-dir /scratch/connectome/.../3.GARD_T1_processed \
        --image-pattern "{subject_id}_brain.nii.gz" \
        --target-name "cognitive status" \
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
from pathlib import Path


DEFAULT_PROMPT = (
    "Analyze this {modality_desc} brain MRI scan. "
    "Estimate the {target_name} of the subject."
)

MODALITY_DESC = {
    'sMRI': 'T1-weighted structural',
    'fMRI': 'functional',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate T1 direct prediction conversations (ablation: no comparison)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_T1_classification_conversations.py \\
      --splits-dir HCvsMCI_splits_300subj \\
      --output-dir HCvsMCI_T1_conversations_300subj \\
      --image-dir /scratch/connectome/.../3.GARD_T1_processed \\
      --image-pattern "{subject_id}_brain.nii.gz" \\
      --target-name "cognitive status" \\
      --modality sMRI
        """
    )
    # Required
    parser.add_argument("--splits-dir", type=str, required=True,
                        help="Directory with *_subjects.csv (from create_T3_comparison_dataset.py)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for conversation JSONL files")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing brain image files")

    # Optional
    parser.add_argument("--image-pattern", type=str, default="{subject_id}.nii.gz",
                        help="Image filename pattern (default: {subject_id}.nii.gz)")
    parser.add_argument("--target-name", type=str, default="target attribute",
                        help="Human-readable target name for prompt "
                             "(e.g. 'cognitive status', 'biological sex')")
    parser.add_argument("--modality", type=str, default="sMRI",
                        choices=["sMRI", "fMRI"],
                        help="Imaging modality (default: sMRI)")
    parser.add_argument("--splits", nargs='+',
                        default=['train', 'validation', 'test'],
                        help="Which splits to generate (default: all)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt template (can use {target_name}, {modality_desc})")
    return parser.parse_args()


def get_image_path(image_dir: str, image_pattern: str, subject_id: str) -> str:
    return f"{image_dir}/{image_pattern.format(subject_id=subject_id)}"


def create_t1_conversation(
    subject_id: str,
    target: str,
    image_path: str,
    prompt: str,
    modality: str,
) -> dict:
    """
    Create a single T1 direct prediction conversation.

    Single-turn: no reference image, no comparison context.
    The model must classify based only on the target image itself.
    """
    return {
        "task_id": f"{subject_id}_direct_prediction",
        "task_type": "T1",
        "subject_ids": [subject_id],
        "modalities": [modality],
        "images": [
            {"path": image_path, "token": "<image>", "modality": modality}
        ],
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "modality": modality, "image_path": image_path},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{target}."}
                ]
            }
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": target,
            "task": "direct_prediction"
        }
    }


def main():
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not splits_dir.exists():
        print(f"Error: Splits directory not found: {splits_dir}")
        return

    # Build prompt
    modality_desc = MODALITY_DESC.get(args.modality, args.modality)
    if args.prompt:
        prompt = args.prompt.format(
            target_name=args.target_name, modality_desc=modality_desc
        )
    else:
        prompt = DEFAULT_PROMPT.format(
            target_name=args.target_name, modality_desc=modality_desc
        )

    print("=" * 60)
    print("T1 DIRECT PREDICTION CONVERSATION GENERATOR (Ablation Exp 1)")
    print("=" * 60)
    print(f"Splits dir:   {splits_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Image dir:    {args.image_dir}")
    print(f"Modality:     {args.modality}")
    print(f"Target name:  {args.target_name}")
    print(f"Prompt:       {prompt}")

    # Process each split
    for split in args.splits:
        subjects_path = splits_dir / f"{split}_subjects.csv"
        if not subjects_path.exists():
            print(f"\nWarning: {subjects_path} not found, skipping {split}.")
            continue

        subjects_df = pd.read_csv(subjects_path)
        conversations = []
        errors = 0

        for _, row in subjects_df.iterrows():
            try:
                subject_id = str(row['subject_id'])
                target = str(row['target'])
                image_path = get_image_path(args.image_dir, args.image_pattern, subject_id)
                conv = create_t1_conversation(
                    subject_id=subject_id,
                    target=target,
                    image_path=image_path,
                    prompt=prompt,
                    modality=args.modality,
                )
                conversations.append(conv)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Warning: {row.get('subject_id', '?')}: {e}")

        jsonl_path = output_dir / f"{split}_conversations.jsonl"
        with open(jsonl_path, 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')

        print(f"\n  {split.upper()}: {len(conversations)} conversations -> {jsonl_path.name}")
        if errors:
            print(f"  Errors: {errors}")

    # Save config
    config = vars(args)
    config['prompt_resolved'] = prompt
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

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
    print(f"      --output-dir ./hf_results/ablation_T1_direct \\")
    print(f"      --eval-output-dir ./eval_predictions_ablation_T1_direct")


if __name__ == "__main__":
    main()
