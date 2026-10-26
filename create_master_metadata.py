"""
Master Clinical Metadata Harmonization
Combines all 146 clinical participants across:
- OpenNeuro ds005907 (Alcohol Use Disorder)
- OpenNeuro ds004504 (Alzheimer's & Dementia)
- PhysioNet CHB-MIT (Epilepsy)
"""

import os
import pandas as pd

BASE_DIR = r"E:\All EEG"
RAW_DIR = os.path.join(BASE_DIR, "Raw_Datasets")

master_records = []

# 1. OpenNeuro ds005907 (Alcoholism)
alc_part_path = os.path.join(RAW_DIR, "openneuro_ds005907_alcohol", "participants.tsv")
if os.path.exists(alc_part_path):
    df_alc = pd.read_csv(alc_part_path, sep='\t')
    for _, row in df_alc.iterrows():
        master_records.append({
            'subject_id': f"ds005907_{row['participant_id']}",
            'original_id': row['participant_id'],
            'dataset': 'ds005907_alcohol',
            'disorder_domain': 'Alcoholism_Addiction',
            'diagnosis': 'AlcoholUseDisorder' if str(row.get('GROUP', '')).strip().upper() == 'AUD' else 'HealthyControl',
            'severity_score_type': 'AUDIT',
            'severity_score': row.get('AUDIT', None),
            'secondary_score': f"BDI={row.get('BDI', 'NA')}",
            'age': row.get('age', None),
            'sex': row.get('sex', None),
            'raw_sampling_rate_hz': 500.0,
            'channels_count': 56,
            'paradigm': 'ThreeArmedBandit_AlcoholCues'
        })

# 2. OpenNeuro ds004504 (Dementia)
dem_part_path = os.path.join(RAW_DIR, "openneuro_ds004504_dementia", "participants.tsv")
if os.path.exists(dem_part_path):
    df_dem = pd.read_csv(dem_part_path, sep='\t')
    for _, row in df_dem.iterrows():
        grp = str(row.get('Group', '')).strip().upper()
        diag_map = {'A': 'AlzheimersDisease', 'F': 'FrontotemporalDementia', 'C': 'HealthyControl'}
        master_records.append({
            'subject_id': f"ds004504_{row['participant_id']}",
            'original_id': row['participant_id'],
            'dataset': 'ds004504_dementia',
            'disorder_domain': 'Neurodegeneration_Dementia',
            'diagnosis': diag_map.get(grp, 'Unknown'),
            'severity_score_type': 'MMSE',
            'severity_score': row.get('MMSE', None),
            'secondary_score': f"Group={grp}",
            'age': row.get('Age', None),
            'sex': row.get('Gender', None),
            'raw_sampling_rate_hz': 500.0,
            'channels_count': 19,
            'paradigm': 'EyesClosedRestingState'
        })

# 3. PhysioNet CHB-MIT (Epilepsy)
sub_info_path = os.path.join(RAW_DIR, "CHB_MIT", "SUBJECT-INFO")
if os.path.exists(sub_info_path):
    with open(sub_info_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        if line.startswith('Case') or line.startswith('#'): continue
        parts = line.split()
        if len(parts) >= 3:
            case, sex, age = parts[0], parts[1], parts[2]
            if case in ['chb01', 'chb02', 'chb03', 'chb04', 'chb05']:
                master_records.append({
                    'subject_id': f"chbmit_{case}",
                    'original_id': case,
                    'dataset': 'chbmit_epilepsy',
                    'disorder_domain': 'Epilepsy_Seizures',
                    'diagnosis': 'PediatricEpilepsy',
                    'severity_score_type': 'SeizureCount',
                    'severity_score': None, # Populated from summary file
                    'secondary_score': 'TelemetryMonitoring',
                    'age': age,
                    'sex': sex,
                    'raw_sampling_rate_hz': 256.0,
                    'channels_count': 23,
                    'paradigm': 'ContinuousClinicalTelemetry'
                })

df_master = pd.DataFrame(master_records)
out_path = os.path.join(BASE_DIR, "master_cohort_metadata.tsv")
df_master.to_csv(out_path, sep='\t', index=False)
print(f"Master metadata compiled successfully with {len(df_master)} clinical subjects.")
print(df_master['disorder_domain'].value_counts())
print("\nDiagnosis Breakdown:")
print(df_master['diagnosis'].value_counts())
