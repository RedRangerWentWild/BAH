import numpy as np
import pandas as pd
import pickle
import lightkurve as lk
from transitleastsquares import transitleastsquares
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Fix multiprocessing on Mac
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# Load trained model
with open('data/model.pkl', 'rb') as f:
    model, le = pickle.load(f)

print("Model loaded successfully")

def extract_features_tls(tic_id):
    try:
        print(f"  Downloading TIC {tic_id}...")
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search) == 0:
            return None, None, "No data found"

        lc = search[0].download()
        lc_clean = lc.remove_outliers(sigma=3).normalize()
        lc_flat = lc_clean.flatten(window_length=401)

        time = lc_flat.time.value
        flux = lc_flat.flux.value
        mask = np.isfinite(time) & np.isfinite(flux)
        time = time[mask]
        flux = flux[mask]

        if len(time) < 100:
            return None, None, "Too few data points"

        print(f"  Running TLS on TIC {tic_id}...")
        tlsmodel = transitleastsquares(time, flux)
        results = tlsmodel.power(
            minimum_period=1,
            maximum_period=15,
            show_progress_bar=False,
            use_threads=1
        )

        depth_ppm = max(0, (1 - results.depth) * 1e6)
        features = {
            'koi_period': results.period,
            'koi_duration': results.duration * 24,
            'koi_depth': depth_ppm,
            'koi_model_snr': results.snr,
            'koi_prad': np.sqrt(depth_ppm / 1e6) * 10,
            'koi_score': min(1.0, results.SDE / 20),
            'koi_fpflag_nt': 0,
            'koi_fpflag_ss': int(results.odd_even_mismatch > 3),
            'koi_fpflag_co': 0,
            'koi_fpflag_ec': 0,
            'koi_impact': 0.15,
            'koi_steff': 5500.0,
            'koi_slogg': 4.5,
            'koi_srad': 1.0,
            'koi_period_err1': results.period * 0.001,
            'koi_depth_err1': depth_ppm * 0.05,
            'koi_duration_err1': results.duration * 24 * 0.03
        }

        return features, results, "OK"

    except Exception as e:
        return None, None, str(e)[:80]


def classify_tess_star(tic_id, plot=True):
    print(f"\n{'='*50}")
    print(f"Processing TIC {tic_id}")
    print('='*50)

    features, tls_results, status = extract_features_tls(tic_id)

    if features is None:
        print(f"  ❌ Failed: {status}")
        return None

    sde = tls_results.SDE
    periodic_dip = sde > 9

    print(f"  SDE: {sde:.2f} ({'significant' if periodic_dip else 'not significant'})")
    print(f"  Period: {features['koi_period']:.4f} days")
    print(f"  Depth: {features['koi_depth']:.1f} ppm")
    print(f"  Duration: {features['koi_duration']:.2f} hours")
    print(f"  Odd/Even mismatch: {tls_results.odd_even_mismatch:.2f}")

    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    pred_label = le.classes_[pred_idx]
    confidence = proba[pred_idx] * 100

    print(f"\n  🎯 Classification: {pred_label.upper()}")
    print(f"  📊 Confidence: {confidence:.1f}%")
    print(f"  🔍 Periodic dip found: {periodic_dip}")

    print("\n  Class probabilities:")
    for cls, prob in sorted(zip(le.classes_, proba), key=lambda x: -x[1]):
        bar = '█' * int(prob * 20)
        print(f"    {cls:<20} {bar} {prob*100:.1f}%")

    if plot and pred_label in ['transit', 'candidate'] and periodic_dip:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        lc_folded_time = (tls_results.folded_phase - 0.5) * features['koi_period']
        axes[0].scatter(lc_folded_time, tls_results.folded_y,
                       s=1, alpha=0.3, color='steelblue')
        axes[0].plot(
            tls_results.model_folded_phase * features['koi_period'] -
            features['koi_period']/2,
            tls_results.model_folded_model,
            'r-', linewidth=2, label='TLS Model'
        )
        axes[0].set_xlabel('Time from Transit Center (days)')
        axes[0].set_ylabel('Normalized Flux')
        axes[0].set_title(f'TIC {tic_id} — Phase Folded Transit')
        axes[0].legend()

        axes[1].plot(tls_results.periods, tls_results.power,
                    color='steelblue', linewidth=0.5)
        axes[1].axvline(
            x=features['koi_period'], color='red',
            linestyle='--',
            label=f"Best period: {features['koi_period']:.3f} days"
        )
        axes[1].set_xlabel('Period (days)')
        axes[1].set_ylabel('SDE Power')
        axes[1].set_title('TLS Periodogram')
        axes[1].legend()

        plt.suptitle(
            f'TIC {tic_id} | {pred_label.upper()} | {confidence:.1f}% confidence',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(f'tls_result_TIC_{tic_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved tls_result_TIC_{tic_id}.png")

    result = {
        'tic_id': tic_id,
        'periodic_dip_found': periodic_dip,
        'classification': pred_label,
        'confidence': round(confidence, 1),
        'period_days': round(features['koi_period'], 4),
        'depth_ppm': round(features['koi_depth'], 1),
        'duration_hours': round(features['koi_duration'], 2),
        'sde': round(sde, 2),
        'odd_even_mismatch': round(tls_results.odd_even_mismatch, 2)
    }

    return result


# Test on 3 known TESS stars
test_stars = [
    259377017,   # TOI-270 - confirmed multi-planet system
    261136679,   # our first test star
    150428135,   # another known planet host
]

results = []
for tic_id in test_stars:
    result = classify_tess_star(tic_id, plot=True)
    if result:
        results.append(result)

print("\n\n" + "="*80)
print("PIPELINE RESULTS SUMMARY")
print("="*80)

if results:
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv('data/tess_results.csv', index=False)
    print("\nSaved to data/tess_results.csv")
else:
    print("No results — all stars failed")