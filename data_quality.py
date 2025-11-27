#!/usr/bin/env python3
"""
Data Quality and Consistency Profiling for PPMI MODEL 1 Dataset

Analyzes missingness, distributions, outliers, temporal consistency,
and cross-domain consistency to determine data quality and reliability.
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from ppmi_config import MODEL1_ROOT, resolve_model1_path


def load_csv_safe(path, **kwargs):
    """Load CSV with error handling for encoding issues."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not decode {path} with any encoding")


def identify_key_columns(df, file_path):
    """Identify key columns (PATNO, EVENT_ID, etc.) for a dataframe."""
    keys = []
    if 'PATNO' in df.columns:
        keys.append('PATNO')
    if 'EVENT_ID' in df.columns:
        keys.append('EVENT_ID')
    if 'REC_ID' in df.columns and len(df['REC_ID'].unique()) == len(df):
        keys.append('REC_ID')
    return keys


def profile_missingness(df, domain_name, file_name):
    """Profile missingness patterns in a dataframe."""
    n_rows = len(df)
    n_cols = len(df.columns)
    
    missing_per_col = df.isnull().sum()
    pct_missing_per_col = (missing_per_col / n_rows * 100).round(2)
    
    # Identify completely missing columns
    completely_missing = missing_per_col[missing_per_col == n_rows].index.tolist()
    
    # Identify columns with >50% missing
    high_missing = missing_per_col[missing_per_col > n_rows * 0.5].index.tolist()
    
    # Per-row missingness
    missing_per_row = df.isnull().sum(axis=1)
    rows_with_any_missing = (missing_per_row > 0).sum()
    rows_completely_missing = (missing_per_row == n_cols).sum()
    
    return {
        'domain': domain_name,
        'file': file_name,
        'n_rows': n_rows,
        'n_cols': n_cols,
        'n_completely_missing_cols': len(completely_missing),
        'n_high_missing_cols': len(high_missing),
        'pct_rows_with_missing': round(rows_with_any_missing / n_rows * 100, 2) if n_rows > 0 else 0,
        'pct_rows_completely_missing': round(rows_completely_missing / n_rows * 100, 2) if n_rows > 0 else 0,
        'avg_missing_per_row': round(missing_per_row.mean(), 2),
        'completely_missing_cols': ','.join(completely_missing[:10]),  # Limit length
        'high_missing_cols': ','.join(high_missing[:10]),
        'missing_details': pd.DataFrame({
            'column': missing_per_col.index,
            'n_missing': missing_per_col.values,
            'pct_missing': pct_missing_per_col.values
        }).sort_values('pct_missing', ascending=False)
    }


def profile_distributions(df, domain_name, file_name):
    """Profile distributions and identify outliers."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_stats = []
    categorical_stats = []
    
    for col in numeric_cols:
        if df[col].notna().sum() == 0:
            continue
        s = df[col].dropna()
        q1, median, q3 = s.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((s < lower_bound) | (s > upper_bound)).sum()
        
        numeric_stats.append({
            'column': col,
            'mean': round(s.mean(), 4),
            'std': round(s.std(), 4),
            'min': s.min(),
            'q25': q1,
            'median': median,
            'q75': q3,
            'max': s.max(),
            'n_outliers': outliers,
            'pct_outliers': round(outliers / len(s) * 100, 2) if len(s) > 0 else 0
        })
    
    for col in categorical_cols[:20]:  # Limit to avoid memory issues
        if df[col].notna().sum() == 0:
            continue
        s = df[col].dropna()
        value_counts = s.value_counts()
        categorical_stats.append({
            'column': col,
            'n_unique': s.nunique(),
            'n_missing': df[col].isnull().sum(),
            'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
            'top_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'top_pct': round(value_counts.iloc[0] / len(s) * 100, 2) if len(value_counts) > 0 else 0
        })
    
    return {
        'domain': domain_name,
        'file': file_name,
        'numeric_stats': pd.DataFrame(numeric_stats) if numeric_stats else pd.DataFrame(),
        'categorical_stats': pd.DataFrame(categorical_stats) if categorical_stats else pd.DataFrame()
    }


def check_temporal_consistency(df, domain_name, file_name):
    """Check temporal consistency using dates and EVENT_ID."""
    results = {
        'domain': domain_name,
        'file': file_name,
        'has_dates': False,
        'has_event_id': 'EVENT_ID' in df.columns,
        'temporal_issues': []
    }
    
    # Check for date columns
    date_cols = [c for c in df.columns if 'DT' in c.upper() or 'DATE' in c.upper()]
    if date_cols:
        results['has_dates'] = True
        results['date_columns'] = date_cols
    
    if 'EVENT_ID' in df.columns and 'PATNO' in df.columns:
        # Check for duplicate PATNO-EVENT_ID combinations
        key_cols = ['PATNO', 'EVENT_ID']
        if 'REC_ID' in df.columns:
            key_cols = ['REC_ID']
        duplicates = df.duplicated(subset=key_cols, keep=False)
        if duplicates.any():
            results['temporal_issues'].append(f"Found {duplicates.sum()} duplicate key combinations")
    
    return results


def check_cross_domain_consistency(demographics_df, other_df, domain_name, file_name):
    """Check consistency of key variables across domains."""
    if 'PATNO' not in demographics_df.columns or 'PATNO' not in other_df.columns:
        return None
    
    results = {
        'domain': domain_name,
        'file': file_name,
        'n_subjects_in_file': other_df['PATNO'].nunique(),
        'n_subjects_in_demographics': demographics_df['PATNO'].nunique(),
        'n_subjects_overlap': len(set(other_df['PATNO'].unique()) & set(demographics_df['PATNO'].unique())),
        'consistency_checks': []
    }
    
    # Check age consistency if available
    if 'BIRTHDT' in demographics_df.columns and 'PATNO' in other_df.columns:
        demo_age = demographics_df[['PATNO', 'BIRTHDT']].drop_duplicates('PATNO')
        merged = other_df[['PATNO']].merge(demo_age, on='PATNO', how='left')
        missing_demo = merged['BIRTHDT'].isnull().sum()
        if missing_demo > 0:
            results['consistency_checks'].append(f"{missing_demo} subjects missing demographic data")
    
    # Check SEX consistency if available
    if 'SEX' in demographics_df.columns and 'PATNO' in other_df.columns:
        demo_sex = demographics_df[['PATNO', 'SEX']].drop_duplicates('PATNO')
        merged = other_df[['PATNO']].merge(demo_sex, on='PATNO', how='left')
        missing_demo = merged['SEX'].isnull().sum()
        if missing_demo > 0:
            results['consistency_checks'].append(f"{missing_demo} subjects missing SEX in demographics")
    
    return results


def determine_consistency_frontier(all_profiles, demographics_df):
    """Determine where data is sufficiently complete and consistent for modeling."""
    if demographics_df is None or 'PATNO' not in demographics_df.columns:
        return None
    
    all_subjects = set(demographics_df['PATNO'].unique())
    
    # Count subjects per domain
    domain_coverage = defaultdict(set)
    domain_visit_coverage = defaultdict(lambda: defaultdict(set))
    
    for profile in all_profiles:
        domain = profile.get('domain', 'Unknown')
        df = profile.get('dataframe')
        if df is None or 'PATNO' not in df.columns:
            continue
        
        subjects = set(df['PATNO'].unique())
        domain_coverage[domain].update(subjects)
        
        if 'EVENT_ID' in df.columns:
            for patno in subjects:
                visits = set(df[df['PATNO'] == patno]['EVENT_ID'].unique())
                domain_visit_coverage[domain][patno].update(visits)
    
    # Build coverage matrix
    coverage_matrix = []
    for domain, subjects in domain_coverage.items():
        coverage_matrix.append({
            'domain': domain,
            'n_subjects': len(subjects),
            'pct_of_all_subjects': round(len(subjects) / len(all_subjects) * 100, 2) if all_subjects else 0,
            'subjects': subjects
        })
    
    return {
        'total_subjects': len(all_subjects),
        'domain_coverage': coverage_matrix,
        'domain_visit_coverage': dict(domain_visit_coverage)
    }


def profile_domain(domain_path, domain_name, demographics_df=None):
    """Profile all files in a domain."""
    domain_profiles = []
    
    if not os.path.isdir(domain_path):
        return domain_profiles
    
    csv_files = list(Path(domain_path).glob('*.csv'))
    
    for csv_file in csv_files:
        try:
            df = load_csv_safe(csv_file)
            file_name = csv_file.name
            
            # Missingness profile
            missing_profile = profile_missingness(df, domain_name, file_name)
            
            # Distribution profile
            dist_profile = profile_distributions(df, domain_name, file_name)
            
            # Temporal consistency
            temporal_profile = check_temporal_consistency(df, domain_name, file_name)
            
            # Cross-domain consistency
            cross_domain_profile = None
            if demographics_df is not None:
                cross_domain_profile = check_cross_domain_consistency(
                    demographics_df, df, domain_name, file_name
                )
            
            domain_profiles.append({
                'domain': domain_name,
                'file': file_name,
                'dataframe': df,
                'missing_profile': missing_profile,
                'dist_profile': dist_profile,
                'temporal_profile': temporal_profile,
                'cross_domain_profile': cross_domain_profile,
                'key_columns': identify_key_columns(df, str(csv_file))
            })
            
        except Exception as e:
            print(f"Error profiling {csv_file}: {e}")
            continue
    
    return domain_profiles


def run_data_quality_analysis(out_dir):
    """Run comprehensive data quality analysis."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Load demographics first for cross-domain checks
    demo_path = resolve_model1_path('Subject_Demographics/Demographics_13Oct2024.csv')
    demographics_df = None
    if demo_path.exists():
        demographics_df = load_csv_safe(str(demo_path))
    
    # Profile each domain
    domains = ['Imaging', 'Motor___MDS-UPDRS', 'Non-motor_Assessments', 
               'Medical_History', 'Subject_Demographics']
    
    all_profiles = []
    missing_summaries = []
    dist_summaries = []
    temporal_summaries = []
    cross_domain_summaries = []
    
    for domain_name in domains:
        domain_path = os.path.join(MODEL1_ROOT, domain_name)
        if not os.path.isdir(domain_path):
            continue
        
        print(f"Profiling domain: {domain_name}")
        domain_profiles = profile_domain(domain_path, domain_name, demographics_df)
        all_profiles.extend(domain_profiles)
        
        # Collect summaries
        for profile in domain_profiles:
            missing_summaries.append(profile['missing_profile'])
            temporal_summaries.append(profile['temporal_profile'])
            if profile['cross_domain_profile']:
                cross_domain_summaries.append(profile['cross_domain_profile'])
    
    # Create summary dataframes
    missing_df = pd.DataFrame(missing_summaries)
    temporal_df = pd.DataFrame(temporal_summaries)
    cross_domain_df = pd.DataFrame(cross_domain_summaries) if cross_domain_summaries else pd.DataFrame()
    
    # Determine consistency frontier
    consistency_frontier = determine_consistency_frontier(all_profiles, demographics_df)
    
    # Save outputs
    missing_df.to_csv(os.path.join(out_dir, 'missingness_summary.csv'), index=False)
    temporal_df.to_csv(os.path.join(out_dir, 'temporal_consistency_summary.csv'), index=False)
    if not cross_domain_df.empty:
        cross_domain_df.to_csv(os.path.join(out_dir, 'cross_domain_consistency_summary.csv'), index=False)
    
    # Save detailed missingness per file
    detailed_missing_dir = os.path.join(out_dir, 'detailed_missingness')
    os.makedirs(detailed_missing_dir, exist_ok=True)
    for profile in all_profiles:
        if not profile['missing_profile']['missing_details'].empty:
            detail_path = os.path.join(
                detailed_missing_dir,
                f"{profile['domain']}_{profile['file'].replace('.csv', '_missingness.csv')}"
            )
            profile['missing_profile']['missing_details'].to_csv(detail_path, index=False)
    
    # Save distribution summaries
    dist_dir = os.path.join(out_dir, 'distributions')
    os.makedirs(dist_dir, exist_ok=True)
    for profile in all_profiles:
        if not profile['dist_profile']['numeric_stats'].empty:
            num_path = os.path.join(
                dist_dir,
                f"{profile['domain']}_{profile['file'].replace('.csv', '_numeric_stats.csv')}"
            )
            profile['dist_profile']['numeric_stats'].to_csv(num_path, index=False)
        
        if not profile['dist_profile']['categorical_stats'].empty:
            cat_path = os.path.join(
                dist_dir,
                f"{profile['domain']}_{profile['file'].replace('.csv', '_categorical_stats.csv')}"
            )
            profile['dist_profile']['categorical_stats'].to_csv(cat_path, index=False)
    
    # Save consistency frontier summary
    if consistency_frontier:
        frontier_df = pd.DataFrame(consistency_frontier['domain_coverage'])
        frontier_df['subjects'] = frontier_df['subjects'].apply(lambda x: len(x) if isinstance(x, set) else 0)
        frontier_df.to_csv(os.path.join(out_dir, 'consistency_frontier_summary.csv'), index=False)
    
    print(f"\nData quality analysis complete. Results saved to {out_dir}")
    print(f"  - Missingness summary: {len(missing_summaries)} files")
    print(f"  - Temporal consistency: {len(temporal_summaries)} files")
    print(f"  - Cross-domain consistency: {len(cross_domain_summaries)} files")
    
    return {
        'missing_summary': missing_df,
        'temporal_summary': temporal_df,
        'cross_domain_summary': cross_domain_df,
        'consistency_frontier': consistency_frontier,
        'all_profiles': all_profiles
    }


def main():
    parser = argparse.ArgumentParser(description='Profile data quality for MODEL 1 dataset')
    parser.add_argument('--out-dir', type=str, default='reports/data_quality',
                        help='Output directory for quality reports')
    args = parser.parse_args()
    
    run_data_quality_analysis(args.out_dir)


if __name__ == '__main__':
    main()

