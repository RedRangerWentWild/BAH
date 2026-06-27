import lightkurve as lk
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

with open('data/model.pkl', 'rb') as f:
    model, le = pickle.load(f)

sector1 = pd.read_csv('data/sector1_tic_ids.csv')
sector1 = sector1.drop_duplicates(subset='TIC ID').reset_index(drop=True)
print(f"Total stars: {len(sector1)}")

OUTPUT_DIR = 'data/tess_sector/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_star(args):
    i, row = args
    tic_id = row['TIC ID']
    disposition = row['TFOPWG Disposition']
    result_file = f"{OUTPUT_DIR}TIC_{tic_id}_result.csv"

    if os.path.exists(result_file):
        saved = pd.read_csv(result_file)
        return f"[{i + 1}] Already done: TIC {tic_id}", saved.iloc[0].to_dict()

    try:
        from transitleastsquares import transitleastsquares

        search = lk.search_lightcurve(f"TIC {tic_id}",
                                      mission="TESS", sector=1)
        if len(search) == 0:
            return f"[{i + 1}] ❌ No data: TIC {tic_id}", None

        lc = search[0].download()
        lc_clean = lc.remove_outliers(sigma=3).normalize()
        lc_flat = lc_clean.flatten(window_length=401)

        time_arr = lc_flat.time.value
        flux_arr = lc_flat.flux.value
        mask = np.isfinite(time_arr) & np.isfinite(flux_arr)
        time_arr = time_arr[mask]
        flux_arr = flux_arr[mask]

        if len(time_arr) < 100:
            return f"[{i + 1}] ❌ Too few points: TIC {tic_id}", None

        tlsmodel = transitleastsquares(time_arr, flux_arr)
        tls_results = tlsmodel.power(
            minimum_period=1,
            maximum_period=15,
            show_progress_bar=False,
            use_threads=1
        )

        depth_ppm = max(0, (1 - tls_results.depth) * 1e6)
        sde = tls_results.SDE
        periodic_dip = sde > 9

        odd_even = tls_results.odd_even_mismatch
        if np.isnan(odd_even):
            odd_even = 0.0

        features = {
            'koi_period': tls_results.period,
            'koi_duration': tls_results.duration * 24,
            'koi_depth': depth_ppm,
            'koi_model_snr': tls_results.snr,
            'koi_prad': np.sqrt(depth_ppm / 1e6) * 10,
            'koi_score': min(1.0, sde / 20),
            'koi_fpflag_nt': 0,
            'koi_fpflag_ss': int(odd_even > 3),
            'koi_fpflag_co': 0,
            'koi_fpflag_ec': 0,
            'koi_impact': 0.15,
            'koi_steff': 5500.0,
            'koi_slogg': 4.5,
            'koi_srad': 1.0,
            'koi_period_err1': tls_results.period * 0.001,
            'koi_depth_err1': depth_ppm * 0.05,
            'koi_duration_err1': tls_results.duration * 24 * 0.03
        }

        X = pd.DataFrame([features])
        proba = model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred_label = le.classes_[pred_idx]
        confidence = proba[pred_idx] * 100

        result = {
            'tic_id': tic_id,
            'true_disposition': disposition,
            'periodic_dip_found': periodic_dip,
            'classification': pred_label,
            'confidence': round(confidence, 1),
            'period_days': round(tls_results.period, 4),
            'depth_ppm': round(depth_ppm, 1),
            'duration_hours': round(tls_results.duration * 24, 2),
            'sde': round(sde, 2)
        }

        pd.DataFrame([result]).to_csv(result_file, index=False)
        return (f"[{i + 1}] ✅ TIC {tic_id} → "
                f"{pred_label} ({confidence:.1f}%) "
                f"SDE:{sde:.1f}"), result

    except Exception as e:
        return f"[{i + 1}] ❌ TIC {tic_id}: {str(e)[:50]}", None


# 3 parallel stars at once - safe on Mac
results = []
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(process_star, args): args
               for args in sector1.iterrows()}

    for future in as_completed(futures):
        msg, result = future.result()
        print(msg)
        if result:
            results.append(result)

        # Save checkpoint every 50
        if len(results) % 50 == 0 and results:
            pd.DataFrame(results).to_csv(
                'data/tess_sector_results.csv', index=False)
            print(f"\n--- Checkpoint saved: {len(results)} done ---\n")

final_df = pd.DataFrame(results)
final_df.to_csv('data/tess_sector_results.csv', index=False)
print(f"\nDONE! {len(results)} stars processed")
print(final_df['classification'].value_counts())