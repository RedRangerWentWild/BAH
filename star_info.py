import pandas as pd

toi = pd.read_csv('toi.csv')

# Map to your 4 classes
label_map = {
    'CP': 'transit',
    'KP': 'transit',
    'PC': 'transit',
    'FP': 'false_positive',
    'FA': 'noise',
    'APC': 'transit'  # treat as transit candidate
}

toi['label'] = toi['TFOPWG Disposition'].map(label_map)

# Keep only the columns we need
dataset = toi[['TIC ID', 'TOI', 'Period (days)', 'Duration (hours)',
               'Depth (mmag)', 'Planet SNR', 'Sectors', 'label']].copy()

dataset = dataset.dropna(subset=['label'])

print(dataset['label'].value_counts())
print(f"\nTotal usable stars: {len(dataset)}")
print(dataset.head())

# Save this as your master label file
dataset.to_csv('data/labels.csv', index=False)
print("\nSaved to data/labels.csv")