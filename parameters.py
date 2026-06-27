import batman
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def estimate_transit_params(period, duration, depth, snr=None, t0=0.0):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = np.sqrt(depth / 1e6)      # depth in ppm → correct ratio
    params.a = 15.0
    params.inc = 90.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.1, 0.3]
    params.limb_dark = "quadratic"

    t = np.linspace(-duration, duration, 1000)
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)

    # Proper confidence from SNR
    if snr is not None and snr > 0:
        # SNR > 10 = high confidence, scales smoothly
        confidence = min(99.0, 50.0 + (snr / 20.0) * 49.0)
    else:
        confidence = 50.0

    return {
        'period_days': round(period, 4),
        'depth_ppm': round(depth, 2),
        'depth_percent': round(depth / 10000, 4),
        'duration_hours': round(duration, 4),
        'planet_radius_ratio': round(params.rp, 4),
        'confidence_percent': round(confidence, 1),
        't': t,
        'flux': flux
    }
def plot_transit_model(results, tic_id="Example"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results['t'], results['flux'], 'b-', linewidth=2, label='Transit Model')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, label='Transit Center')

    # Annotate parameters
    textstr = (
        f"Period: {results['period_days']} days\n"
        f"Depth: {results['depth_ppm']} ppm ({results['depth_percent']}%)\n"
        f"Duration: {results['duration_hours']} hours\n"
        f"Rp/Rs: {results['planet_radius_ratio']}\n"
        f"Confidence: {results['confidence_percent']}%"
    )
    ax.text(0.02, 0.05, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Time from Transit Center (days)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'Transit Model Fit — {tic_id}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'transit_model_{tic_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved transit_model_{tic_id}.png")

# Test on 3 real confirmed planets from our dataset
df = pd.read_csv('data/kepler_labelled.csv')
confirmed = df[df['label'] == 'transit'].dropna().head(3)

for i, row in confirmed.iterrows():
    print(f"\n--- Star {i} ---")
    results = estimate_transit_params(
        period=row['koi_period'],
        duration=row['koi_duration'],
        depth=row['koi_depth'],
        snr=row['koi_model_snr']      # pass SNR now
    )
    print(f"Period:     {results['period_days']} days")
    print(f"Depth:      {results['depth_ppm']} ppm ({results['depth_percent']}%)")
    print(f"Duration:   {results['duration_hours']} hours")
    print(f"Rp/Rs:      {results['planet_radius_ratio']}")
    print(f"Confidence: {results['confidence_percent']}%")
    plot_transit_model(results, tic_id=f"KOI_{i}")