#!/usr/bin/env python3
"""
Model Explainability and Robustness Analysis

Includes:
- Feature importance (tree-based models)
- Permutation importance
- SHAP values (if available)
- Robustness checks
"""

import numpy as np
import pandas as pd
import os
import argparse
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. SHAP analysis will be skipped.")


def get_feature_importance_tree(model, feature_names):
    """Extract feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return importance_df
    return None


def compute_permutation_importance(model, X, y, feature_names, n_repeats=10, random_state=42):
    """Compute permutation importance."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df


def compute_shap_values(model, X, feature_names, max_samples=100):
    """Compute SHAP values if available."""
    if not SHAP_AVAILABLE:
        return None
    
    # Sample if too large
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X
    
    try:
        # Use TreeExplainer for tree models
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class
            if isinstance(shap_values, list):
                # Average across classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            
            shap_df = pd.DataFrame(
                shap_values,
                columns=feature_names
            )
            shap_importance = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            return shap_importance, shap_values
        else:
            # Use KernelExplainer for other models (slower)
            explainer = shap.KernelExplainer(model.predict_proba, X_sample[:50])
            shap_values = explainer.shap_values(X_sample[:20])
            
            if isinstance(shap_values, list):
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            
            shap_importance = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            return shap_importance, shap_values
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return None, None


def robustness_check_feature_subsets(model, X, y, feature_groups, model_name):
    """Test robustness by training on different feature subsets."""
    results = []
    
    for group_name, feature_list in feature_groups.items():
        # Select features
        available_features = [f for f in feature_list if f in X.columns]
        if len(available_features) == 0:
            continue
        
        X_subset = X[available_features]
        
        # Train and evaluate
        model.fit(X_subset, y, model_name=model_name)
        y_pred = model.predict(X_subset, model_name)
        
        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        
        results.append({
            'feature_group': group_name,
            'n_features': len(available_features),
            'accuracy': acc,
            'balanced_accuracy': bal_acc
        })
    
    return pd.DataFrame(results)


def analyze_model_explainability(models_dir, features_csv, out_dir):
    """Comprehensive explainability analysis."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Load features
    features_df = pd.read_csv(features_csv)
    
    # Prepare data
    feature_cols = [c for c in features_df.columns 
                     if c not in ['PATNO', 'diagnosis_label', 'included', 'diagnosis_code', 'baseline_visit']]
    X = features_df[feature_cols].copy()
    y = features_df['diagnosis_label'].copy()
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    # Load trained models (simplified - would need to load from pickle in production)
    from models import BaselineMLModels
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model_wrapper = BaselineMLModels()
    
    # Analyze each model
    model_names = ['random_forest', 'xgboost']
    all_importance = []
    
    for model_name in model_names:
        print(f"\nAnalyzing {model_name}...")
        
        # Train model
        model_wrapper.fit(X_train, y_train, model_name=model_name)
        model = model_wrapper.models[model_name]
        
        # Feature importance (tree-based)
        if hasattr(model, 'feature_importances_'):
            tree_importance = get_feature_importance_tree(model, X.columns)
            if tree_importance is not None:
                tree_importance['model'] = model_name
                tree_importance['method'] = 'feature_importance'
                all_importance.append(tree_importance)
                tree_importance.to_csv(
                    os.path.join(out_dir, f'{model_name}_feature_importance.csv'),
                    index=False
                )
        
        # Permutation importance
        print("  Computing permutation importance...")
        perm_importance = compute_permutation_importance(
            model, X_test, y_test, X.columns, n_repeats=5
        )
        perm_importance['model'] = model_name
        perm_importance['method'] = 'permutation'
        all_importance.append(perm_importance)
        perm_importance.to_csv(
            os.path.join(out_dir, f'{model_name}_permutation_importance.csv'),
            index=False
        )
        
        # SHAP values
        if SHAP_AVAILABLE:
            print("  Computing SHAP values...")
            shap_importance, shap_values = compute_shap_values(
                model, X_test, X.columns, max_samples=50
            )
            if shap_importance is not None:
                shap_importance['model'] = model_name
                shap_importance['method'] = 'shap'
                all_importance.append(shap_importance)
                shap_importance.to_csv(
                    os.path.join(out_dir, f'{model_name}_shap_importance.csv'),
                    index=False
                )
        
        # Robustness checks
        print("  Running robustness checks...")
        feature_groups = {
            'demographics_only': [c for c in X.columns if any(x in c.lower() for x in ['age', 'sex', 'education', 'handed'])],
            'motor_only': [c for c in X.columns if 'updrs' in c.lower() or 'motor' in c.lower()],
            'nonmotor_only': [c for c in X.columns if any(x in c.lower() for x in ['moca', 'gds', 'ess', 'scau', 'upsit'])],
            'imaging_only': [c for c in X.columns if any(x in c.lower() for x in ['sbr', 'volume', 'caudate', 'putamen'])],
            'all_features': list(X.columns)
        }
        
        robustness_results = robustness_check_feature_subsets(
            model_wrapper, X_train, y_train, feature_groups, model_name
        )
        robustness_results['model'] = model_name
        robustness_results.to_csv(
            os.path.join(out_dir, f'{model_name}_robustness.csv'),
            index=False
        )
    
    # Aggregate importance
    if all_importance:
        importance_summary = pd.concat(all_importance, ignore_index=True)
        importance_summary.to_csv(
            os.path.join(out_dir, 'importance_summary.csv'),
            index=False
        )
    
    print(f"\nExplainability analysis complete. Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Model explainability and robustness analysis')
    parser.add_argument('--features-csv', type=str,
                        default='reports/features/baseline_features.csv',
                        help='Path to features CSV')
    parser.add_argument('--models-dir', type=str,
                        default='reports/models/baseline',
                        help='Directory with trained models')
    parser.add_argument('--out-dir', type=str, default='reports/explainability',
                        help='Output directory for explainability results')
    args = parser.parse_args()
    
    analyze_model_explainability(args.models_dir, args.features_csv, args.out_dir)


if __name__ == '__main__':
    main()


