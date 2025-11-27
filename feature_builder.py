#!/usr/bin/env python3
"""
Feature Engineering for PPMI Classification Models

Extracts and engineers features from:
- Demographics
- Motor assessments (MDS-UPDRS)
- Non-motor assessments
- Imaging metrics
- Medical history
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


def extract_demographic_features(demographics_df, event_id='BL'):
    """Extract demographic features at baseline."""
    # Filter to baseline if EVENT_ID available
    if 'EVENT_ID' in demographics_df.columns:
        demo_baseline = demographics_df[demographics_df['EVENT_ID'] == event_id].copy()
    else:
        # If no EVENT_ID, take first record per subject
        demo_baseline = demographics_df.groupby('PATNO').first().reset_index()
    
    features = pd.DataFrame({'PATNO': demo_baseline['PATNO']})
    
    # Age (calculate from birth date if available)
    if 'BIRTHDT' in demo_baseline.columns:
        # Parse birth date (format: MM/YYYY)
        def parse_age(birth_str):
            if pd.isna(birth_str) or birth_str == '':
                return np.nan
            try:
                parts = str(birth_str).split('/')
                if len(parts) == 2:
                    birth_year = int(parts[1])
                    # Approximate baseline year (2010-2015 typical for PPMI)
                    baseline_year = 2012  # Mid-range estimate
                    return baseline_year - birth_year
            except:
                pass
            return np.nan
        
        features['age'] = demo_baseline['BIRTHDT'].apply(parse_age)
    
    # Sex (0=male, 1=female typically)
    if 'SEX' in demo_baseline.columns:
        features['sex'] = demo_baseline['SEX']
    
    # Education (if available)
    if 'EDUCYRS' in demo_baseline.columns:
        features['education_years'] = demo_baseline['EDUCYRS']
    
    # Handedness
    if 'HANDED' in demo_baseline.columns:
        features['handedness'] = demo_baseline['HANDED']
    
    # Race/ethnicity indicators
    race_cols = ['RAASIAN', 'RABLACK', 'RAHAWOPI', 'RAINDALS', 'RANOS', 'RAWHITE']
    for col in race_cols:
        if col in demo_baseline.columns:
            features[col.lower()] = demo_baseline[col]
    
    return features


def extract_motor_features(motor_df, event_id='BL'):
    """Extract motor assessment features at baseline."""
    # Filter to baseline
    motor_baseline = motor_df[motor_df['EVENT_ID'] == event_id].copy()
    
    if len(motor_baseline) == 0:
        return pd.DataFrame()
    
    features = pd.DataFrame({'PATNO': motor_baseline['PATNO']})
    
    # MDS-UPDRS Part I (non-motor experiences)
    if 'NP1TOT' in motor_baseline.columns:
        features['updrs_part1_total'] = pd.to_numeric(motor_baseline['NP1TOT'], errors='coerce')
    
    # MDS-UPDRS Part II (motor experiences of daily living)
    if 'NP2TOT' in motor_baseline.columns:
        features['updrs_part2_total'] = pd.to_numeric(motor_baseline['NP2TOT'], errors='coerce')
    
    # MDS-UPDRS Part III (motor examination) - most important
    if 'NP3TOT' in motor_baseline.columns:
        features['updrs_part3_total'] = pd.to_numeric(motor_baseline['NP3TOT'], errors='coerce')
    
    # Individual Part III items (key motor signs)
    part3_items = [c for c in motor_baseline.columns if c.startswith('NP3') and c != 'NP3TOT']
    for item in part3_items[:20]:  # Limit to avoid too many features
        if motor_baseline[item].dtype in ['object', 'string']:
            features[f'motor_{item.lower()}'] = pd.to_numeric(motor_baseline[item], errors='coerce')
        else:
            features[f'motor_{item.lower()}'] = motor_baseline[item]
    
    # MDS-UPDRS Part IV (motor complications)
    if 'NP4TOT' in motor_baseline.columns:
        features['updrs_part4_total'] = pd.to_numeric(motor_baseline['NP4TOT'], errors='coerce')
    
    # Total UPDRS score (sum of parts)
    updrs_cols = [c for c in features.columns if 'updrs_part' in c and 'total' in c]
    if len(updrs_cols) > 0:
        features['updrs_total'] = features[updrs_cols].sum(axis=1)
    
    # Hoehn & Yahr stage
    if 'NHY' in motor_baseline.columns:
        features['hoehn_yahr'] = pd.to_numeric(motor_baseline['NHY'], errors='coerce')
    
    # Schwab & England ADL
    if 'MSEADLG' in motor_baseline.columns:
        features['schwab_england_adl'] = pd.to_numeric(motor_baseline['MSEADLG'], errors='coerce')
    
    return features


def extract_nonmotor_features(nonmotor_files, event_id='BL'):
    """Extract non-motor assessment features."""
    all_features = []
    
    # Key non-motor assessments to extract
    key_assessments = {
        'Montreal_Cognitive_Assessment__MoCA__13Oct2024.csv': ['MOCATOT'],
        'Geriatric_Depression_Scale__Short_Version__13Oct2024.csv': ['GDSSHORT'],
        'Epworth_Sleepiness_Scale_13Oct2024.csv': ['ESSSCORE'],
        'SCOPA-AUT_13Oct2024.csv': ['SCAUPSY', 'SCAUGAST', 'SCAUGI', 'SCAUCVS', 'SCAUTRS', 'SCAUSEX', 'SCAUTOT'],
        'University_of_Pennsylvania_Smell_Identification_Test_UPSIT_13Oct2024.csv': ['UPSITBK1', 'UPSITBK2', 'UPSITBK3', 'UPSITBK4', 'UPSITTOT'],
        'Trail_Making_A_and_B_13Oct2024.csv': ['TRAA', 'TRAB'],
        'Hopkins_Verbal_Learning_Test_-_Revised_13Oct2024.csv': ['HVLTTR', 'HVLTDR', 'HVLTREC'],
    }
    
    for file_name, cols_to_extract in key_assessments.items():
        file_path = resolve_model1_path(f'Non-motor_Assessments/{file_name}')
        if not file_path.exists():
            continue
        
        try:
            df = load_csv_safe(str(file_path))
            if 'EVENT_ID' in df.columns:
                df_baseline = df[df['EVENT_ID'] == event_id].copy()
            else:
                df_baseline = df.groupby('PATNO').first().reset_index()
            
            if len(df_baseline) == 0:
                continue
            
            file_features = pd.DataFrame({'PATNO': df_baseline['PATNO']})
            
            for col in cols_to_extract:
                if col in df_baseline.columns:
                    file_features[col.lower()] = pd.to_numeric(df_baseline[col], errors='coerce')
            
            if len(file_features.columns) > 1:  # More than just PATNO
                all_features.append(file_features)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    # Merge all non-motor features
    if all_features:
        nonmotor_features = all_features[0]
        for feat_df in all_features[1:]:
            nonmotor_features = nonmotor_features.merge(feat_df, on='PATNO', how='outer')
        return nonmotor_features
    else:
        return pd.DataFrame()


def extract_imaging_features(imaging_files, event_id='BL'):
    """Extract imaging-derived features at baseline."""
    all_features = []
    
    # DaTscan SBR Analysis (key imaging biomarker for PD)
    datscan_path = resolve_model1_path('Imaging/DaTScan_SBR_Analysis_13Oct2024.csv')
    if datscan_path.exists():
        try:
            datscan_df = load_csv_safe(str(datscan_path))
            if 'EVENT_ID' in datscan_df.columns:
                datscan_baseline = datscan_df[datscan_df['EVENT_ID'] == event_id].copy()
            else:
                datscan_baseline = datscan_df.groupby('PATNO').first().reset_index()
            
            if len(datscan_baseline) > 0:
                imaging_features = pd.DataFrame({'PATNO': datscan_baseline['PATNO']})
                
                # SBR values (striatal binding ratio)
                sbr_cols = [c for c in datscan_baseline.columns if 'SBR' in c.upper() or 'CAUDATE' in c.upper() or 'PUTAMEN' in c.upper()]
                for col in sbr_cols[:10]:  # Limit features
                    imaging_features[col.lower()] = pd.to_numeric(datscan_baseline[col], errors='coerce')
                
                all_features.append(imaging_features)
        except Exception as e:
            print(f"Error processing DaTscan: {e}")
    
    # Grey Matter Volume
    gm_path = resolve_model1_path('Imaging/Grey_Matter_Volume_13Oct2024.csv')
    if gm_path.exists():
        try:
            gm_df = load_csv_safe(str(gm_path))
            if 'EVENT_ID' in gm_df.columns:
                gm_baseline = gm_df[gm_df['EVENT_ID'] == event_id].copy()
            else:
                gm_baseline = gm_df.groupby('PATNO').first().reset_index()
            
            if len(gm_baseline) > 0:
                gm_features = pd.DataFrame({'PATNO': gm_baseline['PATNO']})
                
                # Volume columns
                vol_cols = [c for c in gm_baseline.columns if 'VOL' in c.upper() or 'VOLUME' in c.upper()]
                for col in vol_cols[:10]:
                    gm_features[col.lower()] = pd.to_numeric(gm_baseline[col], errors='coerce')
                
                all_features.append(gm_features)
        except Exception as e:
            print(f"Error processing Grey Matter Volume: {e}")
    
    # Merge imaging features
    if all_features:
        imaging_features = all_features[0]
        for feat_df in all_features[1:]:
            imaging_features = imaging_features.merge(feat_df, on='PATNO', how='outer')
        return imaging_features
    else:
        return pd.DataFrame()


def create_derived_features(features_df):
    """Create derived/composite features."""
    derived = features_df.copy()
    
    # Age groups
    if 'age' in derived.columns:
        derived['age_group'] = pd.cut(derived['age'], bins=[0, 50, 60, 70, 100], labels=['<50', '50-60', '60-70', '70+'])
    
    # UPDRS severity categories
    if 'updrs_part3_total' in derived.columns:
        derived['updrs3_severity'] = pd.cut(
            derived['updrs_part3_total'],
            bins=[0, 20, 35, 60, 200],
            labels=['Mild', 'Moderate', 'Severe', 'Very_Severe']
        )
    
    # Cognitive impairment (MoCA < 26)
    if 'mocatot' in derived.columns:
        derived['cognitive_impaired'] = (derived['mocatot'] < 26).astype(int)
    
    # Depression (GDS > 5)
    if 'gdsshort' in derived.columns:
        derived['depression'] = (derived['gdsshort'] > 5).astype(int)
    
    return derived


def build_baseline_feature_set(labels_df, out_dir):
    """Build complete baseline feature set for classification."""
    print("Loading data files...")
    
    # Load demographics
    demo_path = resolve_model1_path('Subject_Demographics/Demographics_13Oct2024.csv')
    demographics_df = load_csv_safe(str(demo_path))
    
    # Load motor assessments
    motor_path = resolve_model1_path('Motor___MDS-UPDRS/MDS-UPDRS_Part_III_13Oct2024.csv')
    motor_df = load_csv_safe(str(motor_path))
    
    # Extract features from each domain
    print("Extracting demographic features...")
    demo_features = extract_demographic_features(demographics_df, event_id='BL')
    
    print("Extracting motor features...")
    motor_features = extract_motor_features(motor_df, event_id='BL')
    
    print("Extracting non-motor features...")
    nonmotor_features = extract_nonmotor_features([], event_id='BL')
    
    print("Extracting imaging features...")
    imaging_features = extract_imaging_features([], event_id='BL')
    
    # Merge all features
    print("Merging features...")
    all_features = labels_df[['PATNO', 'diagnosis_label', 'included']].copy()
    
    # Merge in order of priority
    for feat_df in [demo_features, motor_features, nonmotor_features, imaging_features]:
        if len(feat_df) > 0:
            all_features = all_features.merge(feat_df, on='PATNO', how='left')
    
    # Create derived features
    print("Creating derived features...")
    all_features = create_derived_features(all_features)
    
    # Filter to included subjects
    feature_set = all_features[all_features['included']].copy()
    
    # Save feature set
    feature_set.to_csv(os.path.join(out_dir, 'baseline_features.csv'), index=False)
    
    # Create feature summary
    feature_summary = {
        'domain': ['demographics', 'motor', 'nonmotor', 'imaging', 'derived', 'total'],
        'n_features': [
            len(demo_features.columns) - 1,  # Exclude PATNO
            len(motor_features.columns) - 1 if len(motor_features) > 0 else 0,
            len(nonmotor_features.columns) - 1 if len(nonmotor_features) > 0 else 0,
            len(imaging_features.columns) - 1 if len(imaging_features) > 0 else 0,
            len([c for c in feature_set.columns if c not in all_features.columns]),
            len(feature_set.columns) - 3  # Exclude PATNO, diagnosis_label, included
        ],
        'n_subjects': [
            len(demo_features),
            len(motor_features) if len(motor_features) > 0 else 0,
            len(nonmotor_features) if len(nonmotor_features) > 0 else 0,
            len(imaging_features) if len(imaging_features) > 0 else 0,
            len(feature_set),
            len(feature_set)
        ]
    }
    summary_df = pd.DataFrame(feature_summary)
    summary_df.to_csv(os.path.join(out_dir, 'feature_summary.csv'), index=False)
    
    print(f"\nFeature engineering complete:")
    print(f"  - Total features: {len(feature_set.columns) - 3}")
    print(f"  - Subjects: {len(feature_set)}")
    print(f"  - Missing data: {feature_set.isnull().sum().sum()} total missing values")
    
    return feature_set


def main():
    parser = argparse.ArgumentParser(description='Build features for PPMI classification')
    parser.add_argument('--labels-csv', type=str, 
                        default='reports/target_definition/baseline_diagnosis_labels.csv',
                        help='Path to baseline labels CSV')
    parser.add_argument('--out-dir', type=str, default='reports/features',
                        help='Output directory for features')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    
    # Build feature set
    feature_set = build_baseline_feature_set(labels_df, args.out_dir)
    
    print(f"\nFeatures saved to {args.out_dir}")


if __name__ == '__main__':
    main()


