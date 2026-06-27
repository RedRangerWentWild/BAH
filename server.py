from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import shap
import batman
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('data/model.pkl', 'rb') as f:
    model, le = pickle.load(f)

class InputData(BaseModel):
    period: float
    duration: float
    depth: float
    snr: float
    score: float
    prad: float
    fpflag_ss: int = 0
    fpflag_co: int = 0
    fpflag_nt: int = 0
    fpflag_ec: int = 0

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([{
        'koi_period': data.period,
        'koi_duration': data.duration,
        'koi_depth': data.depth,
        'koi_model_snr': data.snr,
        'koi_prad': data.prad,
        'koi_score': data.score,
        'koi_fpflag_nt': data.fpflag_nt,
        'koi_fpflag_ss': data.fpflag_ss,
        'koi_fpflag_co': data.fpflag_co,
        'koi_fpflag_ec': data.fpflag_ec,
        'koi_impact': 0.15,
        'koi_steff': 5455.0,
        'koi_slogg': 4.5,
        'koi_srad': 0.93,
        'koi_period_err1': data.period * 0.001,
        'koi_depth_err1': data.depth * 0.05,
        'koi_duration_err1': data.duration * 0.03,
    }])

    proba = model.predict_proba(input_df)[0]
    pred_idx = np.argmax(proba)
    pred_label = le.classes_[pred_idx]
    confidence = float(proba[pred_idx] * 100)
    probabilities = {cls: float(p * 100) for cls, p in zip(le.classes_, proba)}

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(input_df)
    transit_idx = list(le.classes_).index('transit')
    shap_arr = shap_vals.values[0, :, transit_idx]
    shap_pairs = sorted(
        zip(input_df.columns.tolist(), shap_arr.tolist()),
        key=lambda x: abs(x[1]), reverse=True
    )[:6]
    shap_out = [{"f": f, "v": round(abs(v), 3), "dir": 1 if v > 0 else -1} for f, v in shap_pairs]

    # Batman transit model
    bp = batman.TransitParams()
    bp.t0 = 0.0
    bp.per = data.period
    bp.rp = float(np.sqrt(data.depth / 1e6))
    bp.a = 15.0; bp.inc = 90.0; bp.ecc = 0.0; bp.w = 90.0
    bp.u = [0.1, 0.3]; bp.limb_dark = "quadratic"
    t_arr = np.linspace(-data.duration, data.duration, 300)
    flux_arr = batman.TransitModel(bp, t_arr).light_curve(bp)

    return {
        "label": pred_label,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "shap": shap_out,
        "params": {
            "period": data.period, "duration": data.duration,
            "depth": data.depth, "snr": data.snr, "prad": data.prad,
            "rprs": round(float(bp.rp), 4),
        },
        "lightcurve": {
            "t": t_arr.tolist(),
            "flux": flux_arr.tolist(),
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}