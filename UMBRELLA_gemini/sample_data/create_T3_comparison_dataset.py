#!/usr/bin/env python3
"""
Generalized T3 Comparison Dataset Creator
==========================================

Creates balanced train/val/test splits and comparison pairs
for any target variable from a metadata CSV.

This replaces the sex-specific 'create_sex_comparison_dataset.py'
with a fully parameterized version.

Usage Example:
    python create_T3_comparison_dataset.py \
        --metadata /path/to/ABCD_phenotype_total.csv \
        --subject-col subjectkey \
        --target-col sex \
        --label-map '{"1":"male","2":"female"}' \
        --output-dir my_comparison_splits \
        --train-size 100 --val-size 100 --test-size 100 \
        --train-same-pairs 40 --train-diff-pairs 40 \
        --val-same-pairs 3 --val-diff-pairs 3 \
        --test-same-pairs 5 --test-diff-pairs 5 \
        --seed 42

Output:
    <output-dir>/
        train_subjects.csv
        validation_subjects.csv
        test_subjects.csv
        train_pairs.csv
        validation_pairs.csv
        test_pairs.csv
        all_subjects_metadata.csv
        config.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create comparison dataset splits and pairs for T3 tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Binary target (sex)
  python create_T3_comparison_dataset.py \\
      --metadata ABCD_phenotype_total.csv \\
      --subject-col subjectkey --target-col sex \\
      --label-map '{"1":"male","2":"female"}' \\
      --output-dir splits_sex_100

  # Multi-class target (diagnosis)
  python create_T3_comparison_dataset.py \\
      --metadata clinical_data.csv \\
      --subject-col patient_id --target-col diagnosis \\
      --output-dir splits_diagnosis_200 \\
      --train-size 200 --val-size 50 --test-size 50
        """
    )

    # --- Data source ---
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to metadata CSV file")
    parser.add_argument("--subject-col", type=str, required=True,
                        help="Column name for subject IDs")
    parser.add_argument("--target-col", type=str, required=True,
                        help="Column name for target variable")

    # --- Label mapping (optional) ---
    parser.add_argument("--label-map", type=str, default=None,
                        help='JSON string for label mapping. '
                             'e.g. \'{"1":"male","2":"female"}\'')

    # --- Output ---
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for splits and pairs")

    # --- Split sizes ---
    parser.add_argument("--train-size", type=int, default=100,
                        help="Total subjects in train split (default: 100)")
    parser.add_argument("--val-size", type=int, default=100,
                        help="Total subjects in validation split (default: 100)")
    parser.add_argument("--test-size", type=int, default=100,
                        help="Total subjects in test split (default: 100)")

    # --- Pair configuration ---
    parser.add_argument("--train-same-pairs", type=int, default=40,
                        help="Same-class pairs per subject in train (default: 40)")
    parser.add_argument("--train-diff-pairs", type=int, default=40,
                        help="Different-class pairs per subject in train (default: 40)")
    parser.add_argument("--val-same-pairs", type=int, default=3,
                        help="Same-class pairs per subject in validation (default: 3)")
    parser.add_argument("--val-diff-pairs", type=int, default=3,
                        help="Different-class pairs per subject in validation (default: 3)")
    parser.add_argument("--test-same-pairs", type=int, default=5,
                        help="Same-class pairs per subject in test (default: 5)")
    parser.add_argument("--test-diff-pairs", type=int, default=5,
                        help="Different-class pairs per subject in test (default: 5)")

    # --- Reproducibility ---
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser.parse_args()


# =============================================================================
# Data Loading
# =============================================================================

def load_metadata(path, subject_col, target_col, label_map=None):
    """
    Load metadata CSV and prepare subject/target columns.

    Args:
        path: Path to CSV file
        subject_col: Column name for subject IDs
        target_col: Column name for target variable
        label_map: Optional JSON string for label mapping

    Returns:
        DataFrame with 'subject_id' and 'target' columns
    """
    print(f"Loading metadata from: {path}")
    df = pd.read_csv(path)
    print(f"  Total rows: {len(df)}")

    # Validate columns exist
    if subject_col not in df.columns:
        raise ValueError(
            f"Subject column '{subject_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Select and rename
    df = df[[subject_col, target_col]].copy()
    df.columns = ['subject_id', 'target']
    df = df.dropna()

    # Apply label mapping if provided
    if label_map:
        mapping = json.loads(label_map)
        df['target'] = df['target'].astype(str).map(mapping)
        unmapped = df['target'].isna().sum()
        if unmapped > 0:
            print(f"  Warning: {unmapped} rows had unmapped values (dropped)")
        df = df.dropna(subset=['target'])
    else:
        df['target'] = df['target'].astype(str)

    print(f"  Valid subjects: {len(df)}")
    print(f"  Target distribution:")
    for label, count in df['target'].value_counts().items():
        print(f"    {label}: {count}")

    return df


# =============================================================================
# Split Creation
# =============================================================================

def create_balanced_splits(df, train_size, val_size, test_size, seed):
    """
    Create balanced train/val/test splits with equal class distribution.

    For N classes, each split has approximately (split_size / N) subjects per class.
    Remainder subjects are distributed round-robin across classes.

    Args:
        df: DataFrame with 'subject_id' and 'target' columns
        train_size: Total subjects in train
        val_size: Total subjects in validation
        test_size: Total subjects in test
        seed: Random seed

    Returns:
        Dictionary {'train': df, 'validation': df, 'test': df}
    """
    classes = sorted(df['target'].unique())
    n_classes = len(classes)

    print(f"\nClasses ({n_classes}): {classes}")

    def distribute_sizes(total, n):
        """Distribute total evenly across n groups, handling remainder."""
        base = total // n
        remainder = total % n
        sizes = [base] * n
        for i in range(remainder):
            sizes[i] += 1
        return sizes

    train_per_class = distribute_sizes(train_size, n_classes)
    val_per_class = distribute_sizes(val_size, n_classes)
    test_per_class = distribute_sizes(test_size, n_classes)

    print(f"\nSampling plan:")
    for i, cls in enumerate(classes):
        pool_size = len(df[df['target'] == cls])
        needed = train_per_class[i] + val_per_class[i] + test_per_class[i]
        print(f"  {cls}: train={train_per_class[i]}, val={val_per_class[i]}, "
              f"test={test_per_class[i]} (need {needed}, have {pool_size})")

    # Sample per class, ensuring no overlap between splits
    splits = {'train': [], 'validation': [], 'test': []}

    for i, cls in enumerate(classes):
        pool = df[df['target'] == cls].copy()
        needed = test_per_class[i] + val_per_class[i] + train_per_class[i]

        if len(pool) < needed:
            raise ValueError(
                f"Class '{cls}': need {needed} subjects, only have {len(pool)}.\n"
                f"Reduce split sizes or check your data."
            )

        # Sample in order: test -> validation -> train (no overlap)
        test_sample = pool.sample(n=test_per_class[i], random_state=seed)
        pool = pool.drop(test_sample.index)

        val_sample = pool.sample(n=val_per_class[i], random_state=seed)
        pool = pool.drop(val_sample.index)

        train_sample = pool.sample(n=train_per_class[i], random_state=seed)

        splits['train'].append(train_sample)
        splits['validation'].append(val_sample)
        splits['test'].append(test_sample)

    # Combine per-class samples and shuffle
    result = {}
    for split_name in splits:
        combined = pd.concat(splits[split_name], ignore_index=True)
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
        combined['split'] = split_name
        result[split_name] = combined

    return result


# =============================================================================
# Pair Creation
# =============================================================================

def create_pairs(splits, split_config, seed):
    """
    Create same-class and different-class comparison pairs for each split.

    For each subject:
    - Same pairs: paired with another subject of the SAME target class
    - Different pairs: paired with a subject of a DIFFERENT target class

    Args:
        splits: Dictionary of split DataFrames
        split_config: Per-split pair counts
        seed: Random seed

    Returns:
        Dictionary of pair DataFrames per split
    """
    np.random.seed(seed)
    all_pairs = {}

    for split_name, df in splits.items():
        n_same = split_config[split_name]['n_same_pairs']
        n_diff = split_config[split_name]['n_diff_pairs']

        print(f"\n{split_name.upper()} pairing:")
        print(f"  Same-class pairs per subject: {n_same}")
        print(f"  Different-class pairs per subject: {n_diff}")

        pairs = []

        for _, row in df.iterrows():
            subject_id = row['subject_id']
            target = row['target']

            # --- Same-class pairs ---
            same_pool = df[
                (df['target'] == target) & (df['subject_id'] != subject_id)
            ]['subject_id'].tolist()

            if same_pool and n_same > 0:
                n_actual = min(n_same, len(same_pool))
                if n_actual < n_same:
                    print(f"  Warning: {subject_id} - only {n_actual} same-class "
                          f"available (requested {n_same})")

                refs = np.random.choice(same_pool, size=n_actual, replace=False)
                for idx, ref in enumerate(refs):
                    ref_target = df[df['subject_id'] == ref]['target'].iloc[0]
                    pairs.append({
                        'subject_id': subject_id,
                        'target': target,
                        'reference_id': ref,
                        'reference_target': ref_target,
                        'comparison_type': 'same',
                        'pair_index': idx,
                        'split': split_name
                    })

            # --- Different-class pairs ---
            diff_pool = df[
                df['target'] != target
            ]['subject_id'].tolist()

            if diff_pool and n_diff > 0:
                n_actual = min(n_diff, len(diff_pool))
                if n_actual < n_diff:
                    print(f"  Warning: {subject_id} - only {n_actual} diff-class "
                          f"available (requested {n_diff})")

                refs = np.random.choice(diff_pool, size=n_actual, replace=False)
                for idx, ref in enumerate(refs):
                    ref_target = df[df['subject_id'] == ref]['target'].iloc[0]
                    pairs.append({
                        'subject_id': subject_id,
                        'target': target,
                        'reference_id': ref,
                        'reference_target': ref_target,
                        'comparison_type': 'different',
                        'pair_index': idx,
                        'split': split_name
                    })

        pairs_df = pd.DataFrame(pairs)
        all_pairs[split_name] = pairs_df
        print(f"  Total pairs: {len(pairs_df)}")

    return all_pairs


# =============================================================================
# Verification & Saving
# =============================================================================

def verify_splits(splits):
    """Print verification summary for all splits."""
    print("\n" + "=" * 60)
    print("SPLIT VERIFICATION")
    print("=" * 60)

    for split_name, df in splits.items():
        print(f"\n{split_name.upper()}: {len(df)} subjects")
        for label, count in df['target'].value_counts().items():
            pct = count / len(df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

    # Check no overlap
    all_ids = []
    for name, df in splits.items():
        ids = set(df['subject_id'].tolist())
        for other_name, other_df in splits.items():
            if name != other_name:
                other_ids = set(other_df['subject_id'].tolist())
                overlap = ids & other_ids
                if overlap:
                    print(f"  WARNING: {len(overlap)} overlapping subjects "
                          f"between {name} and {other_name}")
        all_ids.extend(df['subject_id'].tolist())

    print(f"\nTotal unique subjects: {len(set(all_ids))}")


def save_outputs(splits, all_pairs, output_dir, args):
    """Save all outputs to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # Save split CSVs
    for name, df in splits.items():
        path = output_dir / f"{name}_subjects.csv"
        df.to_csv(path, index=False)
        print(f"  {name}_subjects.csv: {len(df)} subjects")

    # Save combined metadata
    combined = pd.concat(splits.values(), ignore_index=True)
    combined.to_csv(output_dir / "all_subjects_metadata.csv", index=False)
    print(f"  all_subjects_metadata.csv: {len(combined)} subjects")

    # Save pair CSVs
    for name, pdf in all_pairs.items():
        path = output_dir / f"{name}_pairs.csv"
        pdf.to_csv(path, index=False)
        print(f"  {name}_pairs.csv: {len(pdf)} pairs")

    # Save config for reproducibility
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  config.json: saved")

    return output_dir


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("T3 COMPARISON DATASET CREATOR (Generalized)")
    print("=" * 60)
    print(f"Metadata: {args.metadata}")
    print(f"Subject column: {args.subject_col}")
    print(f"Target column: {args.target_col}")
    print(f"Label map: {args.label_map}")
    print(f"Seed: {args.seed}")

    # 1. Load metadata
    df = load_metadata(
        args.metadata,
        args.subject_col,
        args.target_col,
        args.label_map
    )

    # 2. Create balanced splits
    splits = create_balanced_splits(
        df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed
    )

    # 3. Verify
    verify_splits(splits)

    # 4. Create comparison pairs
    split_config = {
        'train': {
            'n_same_pairs': args.train_same_pairs,
            'n_diff_pairs': args.train_diff_pairs,
        },
        'validation': {
            'n_same_pairs': args.val_same_pairs,
            'n_diff_pairs': args.val_diff_pairs,
        },
        'test': {
            'n_same_pairs': args.test_same_pairs,
            'n_diff_pairs': args.test_diff_pairs,
        },
    }

    all_pairs = create_pairs(splits, split_config, args.seed)

    # 5. Save
    output_dir = save_outputs(splits, all_pairs, args.output_dir, args)

    # 6. Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Output: {output_dir.absolute()}")
    print(f"\nNext step:")
    print(f"  python generate_T3_comparison_conversations.py \\")
    print(f"      --splits-dir {args.output_dir} \\")
    print(f"      --output-dir <conversation_output_dir> \\")
    print(f"      --image-dir <path_to_images> \\")
    print(f"      --target-name '<human readable target name>'")


if __name__ == "__main__":
    main()
