import pandas as pd
import lightkurve as lk
import os
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

labels = pd.read_csv('data/labels.csv')

transit = labels[labels['label'] == 'transit'].sample(1000, random_state=42)
false_pos = labels[labels['label'] == 'false_positive'].sample(500, random_state=42)
noise = labels[labels['label'] == 'noise']

sample = pd.concat([transit, false_pos, noise]).reset_index(drop=True)
print(f"Total to download: {len(sample)}")
print(sample['label'].value_counts())
print(f"Already downloaded: {len(os.listdir('data/raw/'))}")
print("Starting...\n")

def download_star(args):
    i, row = args
    tic_id = row['TIC ID']
    label = row['label']
    filename = f"data/raw/TIC_{tic_id}.csv"

    if os.path.exists(filename):
        return f"[{i+1}] Already exists: TIC {tic_id}"

    try:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search) == 0:
            return f"[{i+1}] ❌ No data: TIC {tic_id}"

        lc = search[0].download()
        lc.to_pandas().to_csv(filename, index=False)
        time.sleep(1)
        return f"[{i+1}] ✅ Saved TIC {tic_id} ({label})"

    except OSError:
        # Truncated file - clear cache and retry once
        try:
            cache = os.path.expanduser("~/.cache/lightkurve")
            for root, dirs, files in os.walk(cache):
                for f in files:
                    if str(tic_id) in f:
                        os.remove(os.path.join(root, f))
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
            lc = search[0].download()
            lc.to_pandas().to_csv(filename, index=False)
            time.sleep(1)
            return f"[{i+1}] ✅ Saved TIC {tic_id} (retry worked)"
        except Exception:
            return f"[{i+1}] ❌ Failed TIC {tic_id} even after retry"

    except Exception as e:
        return f"[{i+1}] ❌ Failed TIC {tic_id}: {str(e)[:50]}"

# 2 parallel workers - reliable and easy on NASA servers
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(download_star, args): args for args in sample.iterrows()}