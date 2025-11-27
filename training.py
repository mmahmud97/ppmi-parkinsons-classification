#!/usr/bin/env python3
"""
Training and Evaluation Framework for PPMI Classification Models

Includes:
- Cross-validation
- Model training
- Performance metrics
- Calibration assessment
"""

import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix,
    brier_score_loss, classification_report
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

from models import BaselineMLModels

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelEvaluator:
    """Evaluate models with comprehensive metrics."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_classification(self, y_true, y_pred, y_proba=None, model_name='model'):
        """Evaluate classification performance."""
        results = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        classes = np.unique(y_true)
        
        for i, cls in enumerate(classes):
            results[f'precision_{cls}'] = precision[i]
            results[f'recall_{cls}'] = recall[i]
            results[f'f1_{cls}'] = f1[i]
            results[f'support_{cls}'] = support[i]
        
        # Macro-averaged metrics
        results['precision_macro'] = precision.mean()
        results['recall_macro'] = recall.mean()
        results['f1_macro'] = f1.mean()
        
        # ROC-AUC (multi-class)
        if y_proba is not None:
            try:
                # One-vs-rest AUC
                if len(classes) > 2:
                    results['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='macro'
                    )
                else:
                    results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception as e:
                results['roc_auc_error'] = str(e)
        
        # Brier score
        if y_proba is not None:
            # Convert to binary for brier score (use max probability)
            y_true_binary = (y_true == classes[np.argmax(y_proba.mean(axis=0))]).astype(int)
            y_proba_binary = y_proba.max(axis=1)
            results['brier_score'] = brier_score_loss(y_true_binary, y_proba_binary)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        return results
    
    def cross_validate(self, X, y, model, model_name, n_splits=5, random_state=42):
        """Perform stratified k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        cv_results = []
        fold = 0
        
        for train_idx, val_idx in skf.split(X, y):
            fold += 1
            print(f"  Fold {fold}/{n_splits}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train, model_name=model_name)
            
            # Predict
            y_pred = model.predict(X_val, model_name)
            y_proba = model.predict_proba(X_val, model_name)
            
            # Evaluate
            fold_results = self.evaluate_classification(y_val, y_pred, y_proba, f"{model_name}_fold{fold}")
            fold_results['fold'] = fold
            cv_results.append(fold_results)
        
        # Aggregate results
        cv_summary = self.aggregate_cv_results(cv_results)
        cv_summary['model'] = model_name
        cv_summary['n_folds'] = n_splits
        
        return cv_summary, cv_results
    
    def aggregate_cv_results(self, cv_results):
        """Aggregate cross-validation results."""
        metrics = ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        if 'roc_auc_ovr' in cv_results[0]:
            metrics.append('roc_auc_ovr')
        if 'brier_score' in cv_results[0]:
            metrics.append('brier_score')
        
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in cv_results if metric in r]
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
        
        return summary


def prepare_data(features_df, target_col='diagnosis_label'):
    """Prepare data for modeling."""
    # Separate features and target
    feature_cols = [c for c in features_df.columns 
                     if c not in ['PATNO', target_col, 'included', 'diagnosis_code', 'baseline_visit']]
    
    X = features_df[feature_cols].copy()
    y = features_df[target_col].copy()
    
    # Handle missing values (simple imputation for now)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)  # For any remaining non-numeric columns
    
    # Remove any remaining non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    return X, y


def train_baseline_models(features_csv, out_dir, n_splits=5):
    """Train and evaluate baseline ML models."""
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading features...")
    features_df = pd.read_csv(features_csv)
    
    print("Preparing data...")
    X, y = prepare_data(features_df)
    
    print(f"\nData shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize models
    model_wrapper = BaselineMLModels()
    evaluator = ModelEvaluator()
    
    # Models to train
    model_names = ['logistic_regression', 'random_forest', 'xgboost']
    
    all_results = []
    test_results = []
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Cross-validation
        print(f"\nCross-validation ({n_splits} folds)...")
        cv_summary, cv_folds = evaluator.cross_validate(
            X_train, y_train, model_wrapper, model_name, n_splits=n_splits
        )
        all_results.append(cv_summary)
        
        # Train on full training set
        print("\nTraining on full training set...")
        model_wrapper.fit(X_train, y_train, model_name=model_name)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        y_pred_test = model_wrapper.predict(X_test, model_name)
        y_proba_test = model_wrapper.predict_proba(X_test, model_name)
        
        test_result = evaluator.evaluate_classification(
            y_test, y_pred_test, y_proba_test, f"{model_name}_test"
        )
        test_results.append(test_result)
        
        # Print results
        print(f"\nCV Results:")
        for key, val in cv_summary.items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.4f}")
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_result['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {test_result['balanced_accuracy']:.4f}")
        print(f"  F1 Macro: {test_result['f1_macro']:.4f}")
        if 'roc_auc_ovr' in test_result:
            print(f"  ROC-AUC: {test_result['roc_auc_ovr']:.4f}")
    
    # Save results
    cv_df = pd.DataFrame(all_results)
    cv_df.to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    
    test_df = pd.DataFrame(test_results)
    test_df.to_csv(os.path.join(out_dir, 'test_results.csv'), index=False)
    
    print(f"\nResults saved to {out_dir}")
    
    return cv_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Train baseline ML models')
    parser.add_argument('--features-csv', type=str,
                        default='reports/features/baseline_features.csv',
                        help='Path to features CSV')
    parser.add_argument('--out-dir', type=str, default='reports/models/baseline',
                        help='Output directory for model results')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Number of CV folds')
    args = parser.parse_args()
    
    train_baseline_models(args.features_csv, args.out_dir, args.n_splits)


if __name__ == '__main__':
    main()


