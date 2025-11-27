#!/usr/bin/env python3
"""
Target Definition for PPMI Classification Problems

Defines labels and inclusion criteria for:
1. Primary: Baseline diagnosis/status classification
2. Extended: Conversion and progression tasks
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from ppmi_config import MODEL1_ROOT, resolve_model1_path


def load_csv_safe(path, **kwargs):
    """Load CSV with error handling."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not decode {path}")


def load_demographics():
    """Load demographics and cohort history."""
    demo_path = resolve_model1_path('Subject_Demographics/Demographics_13Oct2024.csv')
    cohort_path = resolve_model1_path('Subject_Demographics/Subject_Cohort_History_13Oct2024.csv')
    
    demographics = load_csv_safe(str(demo_path))
    cohort_history = load_csv_safe(str(cohort_path))
    
    return demographics, cohort_history


def load_clinical_diagnosis():
    """Load clinical diagnosis data."""
    diag_path = resolve_model1_path('Medical_History/Clinical_Diagnosis_13Oct2024.csv')
    return load_csv_safe(str(diag_path))


def explore_diagnosis_codes(cohort_history, clinical_diag):
    """Explore diagnosis codes to understand their meaning."""
    print("=== APPRDX Codes (from Subject_Cohort_History) ===")
    print(cohort_history['APPRDX'].value_counts().sort_index())
    print("\n=== COHORT Codes ===")
    print(cohort_history['COHORT'].value_counts().sort_index())
    
    if clinical_diag is not None and 'NEWDIAG' in clinical_diag.columns:
        print("\n=== NEWDIAG Codes (from Clinical_Diagnosis) ===")
        print(clinical_diag['NEWDIAG'].value_counts().sort_index())
    
    # Common PPMI codes (based on documentation):
    # APPRDX: 1=PD, 2=Control, 3=SWEDD, 4=Prodromal/At-risk
    # COHORT: 1=PD, 2=Control, 3=SWEDD, 4=Prodromal
    print("\n=== Expected PPMI Code Meanings ===")
    print("APPRDX: 1=PD, 2=Control, 3=SWEDD, 4=Prodromal/At-risk")
    print("COHORT: 1=PD, 2=Control, 3=SWEDD, 4=Prodromal")


def define_baseline_diagnosis_labels(cohort_history, clinical_diag=None, event_id='BL'):
    """
    Define baseline diagnosis labels for primary classification task.
    
    Strategy:
    - Use APPRDX from Subject_Cohort_History as primary label source
    - Map to: PD (1), Control (2), SWEDD (3), Prodromal (4)
    - For baseline, use the subject's cohort assignment
    """
    # Start with cohort history (one row per subject)
    labels = cohort_history[['PATNO', 'APPRDX', 'COHORT']].copy()
    
    # Map APPRDX to label names
    label_map = {
        1: 'PD',
        2: 'Control',
        3: 'SWEDD',
        4: 'Prodromal'
    }
    
    labels['diagnosis_label'] = labels['APPRDX'].map(label_map)
    labels['diagnosis_code'] = labels['APPRDX']
    
    # Add baseline visit indicator (all subjects have baseline by definition in cohort history)
    labels['baseline_visit'] = True
    
    # Check for missing or invalid codes
    invalid = labels['diagnosis_label'].isnull()
    if invalid.any():
        print(f"Warning: {invalid.sum()} subjects with invalid APPRDX codes")
        print(labels[invalid][['PATNO', 'APPRDX', 'COHORT']])
    
    return labels


def define_conversion_labels(cohort_history, clinical_diag, min_followup_years=3):
    """
    Define conversion labels for prodromal/at-risk subjects.
    
    Strategy:
    - Identify subjects who start as prodromal/at-risk (APPRDX=4 or COHORT=4)
    - Check if they convert to PD (NEWDIAG=1 or APPRDX changes to 1) within time window
    - Requires longitudinal follow-up data
    """
    # Start with prodromal subjects
    prodromal_subjects = cohort_history[
        (cohort_history['APPRDX'] == 4) | (cohort_history['COHORT'] == 4)
    ]['PATNO'].unique()
    
    if len(prodromal_subjects) == 0:
        print("No prodromal subjects found in cohort history")
        return None
    
    # Check clinical diagnosis for conversion events
    if clinical_diag is None or 'NEWDIAG' not in clinical_diag.columns:
        print("Clinical diagnosis data not available for conversion analysis")
        return None
    
    # Filter to prodromal subjects
    prodromal_diag = clinical_diag[clinical_diag['PATNO'].isin(prodromal_subjects)].copy()
    
    # Identify conversions (NEWDIAG=1 indicates PD diagnosis)
    # Note: Need to check if this is a new diagnosis or confirmation
    conversions = prodromal_diag[prodromal_diag['NEWDIAG'] == 1].copy()
    
    # For now, create a simple conversion indicator
    # In practice, would need to check dates and ensure it's within time window
    conversion_labels = pd.DataFrame({
        'PATNO': prodromal_subjects
    })
    conversion_labels['converted'] = conversion_labels['PATNO'].isin(conversions['PATNO'].unique())
    conversion_labels['conversion_label'] = conversion_labels['converted'].map({True: 'Converted', False: 'Not_Converted'})
    
    return conversion_labels


def define_progression_labels(cohort_history, motor_data_path=None):
    """
    Define progression labels for PD subjects.
    
    Strategy:
    - Identify PD subjects (APPRDX=1)
    - Calculate change in MDS-UPDRS Part III (motor) scores over time
    - Classify as fast vs slow progressors based on rate of change
    """
    pd_subjects = cohort_history[cohort_history['APPRDX'] == 1]['PATNO'].unique()
    
    if len(pd_subjects) == 0:
        print("No PD subjects found")
        return None
    
    # Load MDS-UPDRS Part III (motor examination)
    if motor_data_path is None:
        motor_data_path = resolve_model1_path('Motor___MDS-UPDRS/MDS-UPDRS_Part_III_13Oct2024.csv')
    
    if not motor_data_path.exists():
        print(f"MDS-UPDRS Part III not found at {motor_data_path}")
        return None
    
    motor_df = load_csv_safe(str(motor_data_path))
    
    # Filter to PD subjects and baseline + follow-up visits
    pd_motor = motor_df[motor_df['PATNO'].isin(pd_subjects)].copy()
    
    # Calculate total motor score (sum of relevant items)
    # MDS-UPDRS Part III typically has items like NP3TOT, but structure may vary
    score_cols = [c for c in pd_motor.columns if 'NP3' in c or 'TOT' in c or 'SUM' in c]
    
    if len(score_cols) == 0:
        print("Could not identify motor score columns")
        return None
    
    # Use the first total/sum column found
    score_col = score_cols[0]
    
    # Group by subject and calculate progression metrics
    progression_data = []
    for patno in pd_subjects:
        subject_data = pd_motor[pd_motor['PATNO'] == patno].sort_values('EVENT_ID')
        
        if len(subject_data) < 2:
            continue  # Need at least 2 visits
        
        baseline_score = subject_data.iloc[0][score_col]
        latest_score = subject_data.iloc[-1][score_col]
        
        if pd.isna(baseline_score) or pd.isna(latest_score):
            continue
        
        # Calculate change and rate (simplified - would need actual dates for proper rate)
        score_change = latest_score - baseline_score
        n_visits = len(subject_data)
        
        progression_data.append({
            'PATNO': patno,
            'baseline_score': baseline_score,
            'latest_score': latest_score,
            'score_change': score_change,
            'n_visits': n_visits
        })
    
    if len(progression_data) == 0:
        print("No progression data available")
        return None
    
    progression_df = pd.DataFrame(progression_data)
    
    # Classify as fast vs slow progressors (median split or clinical threshold)
    median_change = progression_df['score_change'].median()
    progression_df['progression_label'] = progression_df['score_change'].apply(
        lambda x: 'Fast_Progressor' if x >= median_change else 'Slow_Progressor'
    )
    
    return progression_df


def create_inclusion_criteria_baseline(labels_df, demographics_df, min_age=None, require_demographics=True):
    """
    Define inclusion/exclusion criteria for baseline classification task.
    
    Criteria:
    - Valid diagnosis label (not missing)
    - Age requirements (if specified)
    - Complete demographics (if required)
    """
    included = pd.Series(True, index=labels_df.index)
    
    # Must have valid diagnosis
    included &= labels_df['diagnosis_label'].notna()
    
    # Age criteria (if specified)
    if min_age is not None and 'BIRTHDT' in demographics_df.columns:
        # Calculate age at baseline (simplified - would need actual baseline date)
        # For now, just check if birth date exists
        demo_subjects = set(demographics_df['PATNO'].unique())
        included &= labels_df['PATNO'].isin(demo_subjects)
    
    # Require demographics
    if require_demographics:
        demo_subjects = set(demographics_df['PATNO'].unique())
        included &= labels_df['PATNO'].isin(demo_subjects)
    
    labels_df['included'] = included
    
    return labels_df


def run_target_definition(out_dir):
    """Run target definition for all classification tasks."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    demographics, cohort_history = load_demographics()
    clinical_diag = load_clinical_diagnosis()
    
    # Explore diagnosis codes
    print("\n=== Exploring Diagnosis Codes ===")
    explore_diagnosis_codes(cohort_history, clinical_diag)
    
    # Define baseline diagnosis labels
    print("\n=== Defining Baseline Diagnosis Labels ===")
    baseline_labels = define_baseline_diagnosis_labels(cohort_history, clinical_diag)
    baseline_labels = create_inclusion_criteria_baseline(baseline_labels, demographics)
    
    print(f"\nBaseline labels distribution:")
    print(baseline_labels['diagnosis_label'].value_counts())
    print(f"\nIncluded subjects: {baseline_labels['included'].sum()} / {len(baseline_labels)}")
    
    # Save baseline labels
    baseline_labels.to_csv(os.path.join(out_dir, 'baseline_diagnosis_labels.csv'), index=False)
    
    # Define conversion labels (if applicable)
    print("\n=== Defining Conversion Labels ===")
    conversion_labels = define_conversion_labels(cohort_history, clinical_diag)
    if conversion_labels is not None:
        print(f"\nConversion labels distribution:")
        print(conversion_labels['conversion_label'].value_counts())
        conversion_labels.to_csv(os.path.join(out_dir, 'conversion_labels.csv'), index=False)
    
    # Define progression labels (if applicable)
    print("\n=== Defining Progression Labels ===")
    progression_labels = define_progression_labels(cohort_history)
    if progression_labels is not None:
        print(f"\nProgression labels distribution:")
        print(progression_labels['progression_label'].value_counts())
        progression_labels.to_csv(os.path.join(out_dir, 'progression_labels.csv'), index=False)
    
    # Create summary
    summary = {
        'task': ['baseline_diagnosis', 'conversion', 'progression'],
        'n_subjects': [
            len(baseline_labels[baseline_labels['included']]),
            len(conversion_labels) if conversion_labels is not None else 0,
            len(progression_labels) if progression_labels is not None else 0
        ],
        'status': [
            'defined',
            'defined' if conversion_labels is not None else 'insufficient_data',
            'defined' if progression_labels is not None else 'insufficient_data'
        ]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, 'target_definition_summary.csv'), index=False)
    
    print(f"\nTarget definition complete. Results saved to {out_dir}")
    
    return {
        'baseline_labels': baseline_labels,
        'conversion_labels': conversion_labels,
        'progression_labels': progression_labels
    }


def main():
    parser = argparse.ArgumentParser(description='Define classification targets for PPMI data')
    parser.add_argument('--out-dir', type=str, default='reports/target_definition',
                        help='Output directory for target definitions')
    args = parser.parse_args()
    
    run_target_definition(args.out_dir)


if __name__ == '__main__':
    main()


