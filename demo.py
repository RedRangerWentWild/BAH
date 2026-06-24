import pandas as pd

kepler = pd.read_csv('cumulative.csv')

# Map to our classes
def classify(row):
    if row['koi_disposition'] == 'CONFIRMED':
        return 'transit'
    elif row['koi_disposition'] == 'FALSE POSITIVE':
        if row['koi_fpflag_ss'] == 1:
            return 'eclipsing_binary'
        elif row['koi_fpflag_co'] == 1:
            return 'blend'
        else:
            return 'false_positive'
    else:
        return 'candidate'

kepler['label'] = kepler.apply(classify, axis=1)
print(kepler['label'].value_counts())

# Keep only useful feature columns
features = [
    'koi_period', 'koi_duration', 'koi_depth',
    'koi_model_snr', 'koi_prad', 'koi_score',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_impact', 'koi_steff', 'koi_slogg', 'koi_srad',
    'koi_period_err1', 'koi_depth_err1', 'koi_duration_err1',
    'label'
]

df = kepler[features].dropna()
print(f"\nClean dataset: {len(df)} stars")
print(df['label'].value_counts())

df.to_csv('data/kepler_labelled.csv', index=False)
print("\nSaved to data/kepler_labelled.csv")