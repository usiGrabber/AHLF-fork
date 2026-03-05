#!/usr/bin/env python
"""
Combine predictions from the AHLFp model ensemble (alpha, beta, gamma, delta)
by calculating the arithmetic mean of prediction scores, then evaluate
balanced accuracy on phospho and non-phospho spectra.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, f1_score, roc_auc_score
import argparse
import os

def load_and_combine_predictions(results_dir, prefix):
    """
    Load predictions from all four models and combine them using arithmetic mean.
    
    Args:
        results_dir: Directory containing the TSV result files
        prefix: Either 'phospho' or 'non_phospho'
    
    Returns:
        DataFrame with combined predictions
    """
    
    # Load all model predictions
    dfs = []
    filename = os.path.join(results_dir, f'{prefix}.tsv')
    df = pd.read_csv(filename, sep='\t', index_col=0)
    dfs.append(df)

    # Ensure all files have the same length
    lengths = [len(df) for df in dfs]
    assert all(length == lengths[0] for length in lengths), "All model result files must have the same number of rows."

    # Combine predictions by row order
    combined = pd.DataFrame()
    combined['ensemble_score'] = np.mean([df['score'].values for df in dfs], axis=0)
    combined['ensemble_pred'] = (combined['ensemble_score'] >= 0.5).astype(float)

    return combined

def calculate_metrics(df, true_label, label_name):
    """
    Calculate and print performance metrics.
    
    Args:
        df: DataFrame with predictions
        true_label: The true label (1 for phospho, 0 for non-phospho)
        label_name: Name for printing (e.g., 'Phospho')
    """
    y_true = np.full(len(df), true_label)
    y_pred = df['ensemble_pred'].values
    
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n{label_name} Spectra Results:")
    print(f"  Total spectra: {len(df)}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Mean ensemble score: {df['ensemble_score'].mean():.4f}")
    print(f"  Std ensemble score: {df['ensemble_score'].std():.4f}")
    print(f"  Correctly classified: {int(acc * len(df))}/{len(df)}")
    
    # Show confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    Predicted 0: {cm[0] if len(cm) > 0 else 'N/A'}")
    if len(cm) > 1:
        print(f"    Predicted 1: {cm[1]}")
    
    return y_true, y_pred, acc

def main():
    parser = argparse.ArgumentParser(
        description='Combine ensemble predictions and calculate balanced accuracy'
    )
    parser.add_argument(
        'results_dir', 
        type=str, 
        help='Directory containing the TSV result files from all four models'
    )
    parser.add_argument(
        '--phospho-prefix', 
        type=str, 
        default='phospho',
        help='Prefix for phospho files (default: phospho)'
    )
    parser.add_argument(
        '--non-phospho-prefix', 
        type=str, 
        default='non_phospho',
        help='Prefix for non-phospho files (default: non_phospho)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='ensemble_combined_results.tsv',
        help='Output file for combined results (default: ensemble_combined_results.tsv)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("AHLFp Ensemble Model Evaluation")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    
    # Combine phospho predictions
    print("\nCombining phospho predictions...")
    phospho_df = load_and_combine_predictions(args.results_dir, args.phospho_prefix)
    phospho_df['true_label'] = 1
    
    # Combine non-phospho predictions
    print("Combining non-phospho predictions...")
    non_phospho_df = load_and_combine_predictions(args.results_dir, args.non_phospho_prefix)
    non_phospho_df['true_label'] = 0
    
    # Calculate metrics for each class
    y_true_phospho, y_pred_phospho, acc_phospho = calculate_metrics(
        phospho_df, 1, "Phospho"
    )
    y_true_non_phospho, y_pred_non_phospho, acc_non_phospho = calculate_metrics(
        non_phospho_df, 0, "Non-Phospho"
    )
    
    # Calculate balanced accuracy
    y_true_all = np.concatenate([y_true_phospho, y_true_non_phospho])
    y_pred_all = np.concatenate([y_pred_phospho, y_pred_non_phospho])
    y_score_all = np.concatenate([phospho_df['ensemble_score'], non_phospho_df['ensemble_score']])

    balanced_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_f1 = f1_score(y_true_all, y_pred_all)
    overall_roc_auc = roc_auc_score(y_true_all, y_score_all)

    print("\n" + "="*70)
    print("Overall Performance Metrics:")
    print("="*70)
    print(f"Total spectra: {len(y_true_all)}")
    print(f"Phospho spectra: {len(y_true_phospho)}")
    print(f"Non-phospho spectra: {len(y_true_non_phospho)}")
    print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall ROC-AUC: {overall_roc_auc:.4f}")
    print(f"Phospho Accuracy: {acc_phospho:.4f}")
    print(f"Non-Phospho Accuracy: {acc_non_phospho:.4f}")

    # Overall confusion matrix
    cm_overall = confusion_matrix(y_true_all, y_pred_all)
    print(f"\nOverall Confusion Matrix:")
    print(f"                Predicted")
    print(f"              0           1")
    print(f"Actual 0  {cm_overall[0][0]:6d}  {cm_overall[0][1]:6d}")
    print(f"       1  {cm_overall[1][0]:6d}  {cm_overall[1][1]:6d}")

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm_overall.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nSensitivity (Recall for phospho): {sensitivity:.4f}")
    print(f"Specificity (Recall for non-phospho): {specificity:.4f}")

    # Save combined results
    combined_all = pd.concat([phospho_df, non_phospho_df], ignore_index=True)
    combined_all.to_csv(args.output, sep='\t')
    print(f"\nCombined results saved to: {args.output}")
    print("="*70)

if __name__ == '__main__':
    main()
