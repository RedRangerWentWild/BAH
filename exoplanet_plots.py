


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

DATA_PATH  = "BAH/data/kepler_labelled.csv"  
OUTPUT_DIR = "plotly_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


COLOR_MAP = {
    "transit":          "#1D9E75",
    "eclipsing_binary": "#534AB7",
    "blend":            "#D85A30",
    "false_positive":   "#BA7517",
    "candidate":        "#185FA5",
}
LABEL_NAMES = {
    "transit":          "Exoplanet Transit",
    "eclipsing_binary": "Eclipsing Binary",
    "blend":            "Stellar Blend",
    "false_positive":   "False Positive",
    "candidate":        "Candidate",
}
ORDER = ["transit", "eclipsing_binary", "blend", "false_positive", "candidate"]

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Arial, sans-serif", color="#3d3d3a", size=13),
    margin=dict(l=60, r=40, t=65, b=60),
)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  ✓  Saved → {path}")
    fig.show()



df = pd.read_csv(DATA_PATH)



print("\n[1/8] Class distribution")

vc     = df["label"].value_counts()
labels = [l for l in ORDER if l in vc.index]
counts = [vc[l] for l in labels]
names  = [LABEL_NAMES[l] for l in labels]
colors = [COLOR_MAP[l]   for l in labels]
total  = sum(counts)

fig1 = go.Figure(go.Bar(
    x=counts, y=names,
    orientation="h",
    marker_color=colors,
    text=[f"{c:,}" for c in counts],
    textposition="outside",
    textfont=dict(size=13, color="#3d3d3a"),
    hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
))


for i, (n, c) in enumerate(zip(names, counts)):
    fig1.add_annotation(
        x=c + max(counts) * 0.01, y=n,
        text=f"  {c / total * 100:.1f}%",
        showarrow=False, xanchor="left",
        font=dict(size=11, color="#888780"),
    )

fig1.update_layout(
    **LAYOUT_BASE,
    title=dict(text="Dataset class distribution — 7,904 Kepler stars",
               x=0.02, font=dict(size=17, color="#2C2C2A")),
    xaxis=dict(title="Number of stars", showgrid=True,
               gridcolor="#E8E6DF", gridwidth=0.5, zeroline=False,
               range=[0, max(counts) * 1.22]),
    yaxis=dict(showgrid=False, autorange="reversed"),
    showlegend=False,
    height=370,
)
save(fig1, "1_class_distribution")

═
print("\n[2/8] Confusion matrix")

CLASS_NAMES = ["Blend", "Candidate", "Ecl. Binary", "False Pos.", "Transit"]
CM_RAW = np.array([
    [211,   1,   0,   0,   0],
    [  0, 272,   3,   0,  81],
    [  0,   0, 412,   0,   0],
    [  0,   0,   0, 145,   0],
    [  0,  61,   0,   3, 392],
])
row_sums = CM_RAW.sum(axis=1, keepdims=True)
cm_pct   = np.where(row_sums > 0, CM_RAW / row_sums * 100, 0)

text_matrix = [
    [f"{CM_RAW[i,j]}<br>{cm_pct[i,j]:.1f}%" for j in range(5)]
    for i in range(5)
]

fig2 = go.Figure(go.Heatmap(
    z=cm_pct,
    x=CLASS_NAMES,
    y=CLASS_NAMES,
    text=text_matrix,
    texttemplate="%{text}",
    textfont=dict(size=12),
    colorscale=[[0, "#EAF3DE"], [0.5, "#378ADD"], [1, "#042C53"]],
    showscale=True,
    colorbar=dict(title="Recall %", tickfont=dict(size=11)),
    hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br><extra></extra>",
))
fig2.update_layout(
    **LAYOUT_BASE,
    title=dict(text="Confusion matrix — XGBoost classifier",
               x=0.02, font=dict(size=17, color="#2C2C2A")),
    xaxis=dict(title="Predicted", side="bottom"),
    yaxis=dict(title="Actual", autorange="reversed"),
    height=470,
)
save(fig2, "2_confusion_matrix")



print("\n[3/8] Per-class metrics")


cm_np = CM_RAW.astype(float)
precision = [cm_np[i,i] / cm_np[:,i].sum() if cm_np[:,i].sum() > 0 else 0 for i in range(5)]
recall    = [cm_np[i,i] / cm_np[i,:].sum() if cm_np[i,:].sum() > 0 else 0 for i in range(5)]
f1        = [2*p*r/(p+r) if p+r > 0 else 0 for p,r in zip(precision, recall)]

metric_colors = {"Precision": "#185FA5", "Recall": "#1D9E75", "F1 Score": "#D85A30"}

fig3 = go.Figure()
for vals, name in [(precision, "Precision"), (recall, "Recall"), (f1, "F1 Score")]:
    fig3.add_trace(go.Bar(
        name=name, x=CLASS_NAMES, y=vals,
        marker_color=metric_colors[name],
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=11),
    ))

fig3.update_layout(
    **LAYOUT_BASE,
    title=dict(text="Per-class model performance",
               x=0.02, font=dict(size=17, color="#2C2C2A")),
    barmode="group",
    yaxis=dict(title="Score", range=[0, 1.12],
               showgrid=True, gridcolor="#E8E6DF", gridwidth=0.5, zeroline=False),
    xaxis=dict(showgrid=False),
    legend=dict(orientation="h", y=1.02, x=1.0, xanchor="right", yanchor="bottom"),
    height=420,
)
save(fig3, "3_per_class_metrics")


print("\n[4/8] Feature importance")

FEAT_LABELS = {
    "koi_fpflag_ss":      "FP: secondary eclipse",
    "koi_fpflag_co":      "FP: centroid offset",
    "koi_fpflag_nt":      "FP: not transit",
    "koi_score":          "KOI disposition score",
    "koi_fpflag_ec":      "FP: ephemeris contam.",
    "koi_model_snr":      "Model SNR",
    "koi_impact":         "Impact parameter",
    "koi_prad":           "Planet radius",
    "koi_duration":       "Transit duration",
    "koi_period":         "Orbital period",
    "koi_depth_err1":     "Depth uncertainty",
    "koi_duration_err1":  "Duration uncertainty",
    "koi_slogg":          "Stellar log g",
    "koi_period_err1":    "Period uncertainty",
    "koi_depth":          "Transit depth",
    "koi_srad":           "Stellar radius",
    "koi_steff":          "Stellar Teff",
}

feat_full = {
    "koi_fpflag_ss": 0.524, "koi_fpflag_co": 0.187, "koi_fpflag_nt": 0.178,
    "koi_score": 0.047, "koi_fpflag_ec": 0.022, "koi_model_snr": 0.013,
    "koi_impact": 0.007, "koi_prad": 0.006, "koi_duration": 0.005,
    "koi_period": 0.004, "koi_depth_err1": 0.003, "koi_duration_err1": 0.003,
    "koi_slogg": 0.003, "koi_period_err1": 0.003, "koi_depth": 0.002,
    "koi_srad": 0.001, "koi_steff": 0.001,
}
feat_phys = {
    "koi_score": 0.490, "koi_depth": 0.079, "koi_prad": 0.076,
    "koi_model_snr": 0.073, "koi_period_err1": 0.046, "koi_duration_err1": 0.037,
    "koi_period": 0.028, "koi_duration": 0.027, "koi_impact": 0.026,
    "koi_depth_err1": 0.025, "koi_slogg": 0.023, "koi_srad": 0.023, "koi_steff": 0.022,
}

fig4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Full model (with FP flags)", "Physics-only model"],
    horizontal_spacing=0.14,
)

sf = sorted(feat_full.items(), key=lambda x: x[1])
fig4.add_trace(go.Bar(
    x=[v for k, v in sf],
    y=[FEAT_LABELS.get(k, k) for k, v in sf],
    orientation="h", marker_color="#185FA5",
    text=[f"{v:.3f}" for k, v in sf],
    textposition="outside", textfont=dict(size=9),
    name="Full model", showlegend=False,
), row=1, col=1)

sp = sorted(feat_phys.items(), key=lambda x: x[1])
fig4.add_trace(go.Bar(
    x=[v for k, v in sp],
    y=[FEAT_LABELS.get(k, k) for k, v in sp],
    orientation="h", marker_color="#1D9E75",
    text=[f"{v:.3f}" for k, v in sp],
    textposition="outside", textfont=dict(size=9),
    name="Physics model", showlegend=False,
), row=1, col=2)

fig4.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Arial, sans-serif", color="#3d3d3a", size=12),
    margin=dict(l=200, r=60, t=80, b=50),
    title=dict(text="Feature importance: full model vs. physics-only",
               x=0.02, font=dict(size=17, color="#2C2C2A")),
    height=530,
)
fig4.update_xaxes(showgrid=True, gridcolor="#E8E6DF", gridwidth=0.5, zeroline=False)
fig4.update_yaxes(showgrid=False)
save(fig4, "4_feature_importance")



print("\n[5/8] Transit depth violin")

fig5 = go.Figure()
for label in ORDER:
    sub   = df[df["label"] == label]["koi_depth"]
    p95   = sub.quantile(0.95)
    clipped = sub[sub <= p95]
    fig5.add_trace(go.Violin(
        y=clipped,
        name=LABEL_NAMES[label],
        fillcolor=COLOR_MAP[label],
        line_color=COLOR_MAP[label],
        opacity=0.82,
        box_visible=True,
        meanline_visible=True,
        points=False,
        hovertemplate="<b>%{x}</b><br>Depth: %{y:.0f} ppm<extra></extra>",
    ))

fig5.update_layout(
    **LAYOUT_BASE,
    title=dict(text="Transit depth by signal class — log scale, clipped at 95th pct",
               x=0.02, font=dict(size=17, color="#2C2C2A")),
    yaxis=dict(title="Transit depth (ppm)", showgrid=True,
               gridcolor="#E8E6DF", gridwidth=0.5, type="log"),
    xaxis=dict(showgrid=False),
    showlegend=False,
    height=440,
    violingap=0.25,
)
save(fig5, "5_transit_depth_violin")


print("\n[6/8] Period vs depth scatter")

sample6 = pd.concat([
    df[df["label"] == l].sample(min(300, len(df[df["label"] == l])), random_state=42)
    for l in ORDER
])
sample6 = sample6[(sample6["koi_period"] < 200) & (sample6["koi_depth"] < 50000)]

fig6 = go.Figure()
for label in ORDER:
    sub = sample6[sample6["label"] == label]
    fig6.add_trace(go.Scatter(
        x=sub["koi_period"], y=sub["koi_depth"],
        mode="markers",
        name=LABEL_NAMES[label],
        marker=dict(color=COLOR_MAP[label], size=5, opacity=0.72,
                    line=dict(width=0.4, color="white")),
        hovertemplate=(f"<b>{LABEL_NAMES[label]}</b><br>"
                       "Period: %{x:.2f} d<br>Depth: %{y:.0f} ppm<extra></extra>"),
    ))

fig6.update_layout(
    **LAYOUT_BASE,
    title=dict(text="Orbital period vs. transit depth — signal class separation",
               x=0.02, font=dict(size=17, color="#2C2C2A")),
    xaxis=dict(title="Orbital period (days)", showgrid=True,
               gridcolor="#E8E6DF", gridwidth=0.5, type="log"),
    yaxis=dict(title="Transit depth (ppm)", showgrid=True,
               gridcolor="#E8E6DF", gridwidth=0.5, type="log"),
    legend=dict(orientation="v", x=0.99, y=0.99, xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#D3D1C7", borderwidth=0.5),
    height=450,
)
save(fig6, "6_period_vs_depth_scatter")


# — KOI score stacked histogram

print("\n[7/8] KOI score histogram")

bins = np.linspace(0, 1, 21) 

fig7 = go.Figure()
for label in ORDER:
    sub = df[df["label"] == label]["koi_score"]
    hist, edges = np.histogram(sub, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    fig7.add_trace(go.Bar(
        x=centers, y=hist,
        name=LABEL_NAMES[label],
        marker_color=COLOR_MAP[label],
        opacity=0.92,
        width=0.045,
        hovertemplate=(f"<b>{LABEL_NAMES[label]}</b><br>"
                       "Score: %{x:.2f}<br>Count: %{y}<extra></extra>"),
    ))

fig7.update_layout(
    **LAYOUT_BASE,
    title=dict(
        text="KOI score distribution — planets score ~1.0, false positives score ~0",
        x=0.02, font=dict(size=17, color="#2C2C2A")),
    barmode="stack",
    xaxis=dict(title="KOI disposition score (0 = not planet · 1 = planet)",
               showgrid=False, tickformat=".1f"),
    yaxis=dict(title="Number of stars", showgrid=True,
               gridcolor="#E8E6DF", gridwidth=0.5, zeroline=False),
    legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    height=420,
)
save(fig7, "7_koi_score_histogram")



print("\n[8/8] Planet radius vs SNR bubble")

sample8 = pd.concat([
    df[df["label"] == l].sample(min(200, len(df[df["label"] == l])), random_state=7)
    for l in ORDER
])
sample8 = sample8[
    (sample8["koi_prad"] < 30) & (sample8["koi_model_snr"] < 500)
].copy()

# Bubble size: scale transit depth to 4–18 px
depth_cap = sample8["koi_depth"].quantile(0.90)
sample8["bubble_size"] = (
    sample8["koi_depth"].clip(upper=depth_cap) / depth_cap * 14 + 4
)

fig8 = go.Figure()
for label in ORDER:
    sub = sample8[sample8["label"] == label]
    fig8.add_trace(go.Scatter(
        x=sub["koi_model_snr"],
        y=sub["koi_prad"],
        mode="markers",
        name=LABEL_NAMES[label],
        marker=dict(
            color=COLOR_MAP[label],
            size=sub["bubble_size"],
            sizemode="diameter",
            opacity=0.75,
            line=dict(width=0.4, color="white"),
        ),
        hovertemplate=(
            f"<b>{LABEL_NAMES[label]}</b><br>"
            "SNR: %{x:.1f}<br>"
            "Planet radius: %{y:.2f} R⊕<extra></extra>"
        ),
    ))

# Reference lines for Earth and Jupiter radius
for yref, label_txt, dash in [(11.2, "Jupiter (11.2 R⊕)", "dot"), (1.0, "Earth (1.0 R⊕)", "dash")]:
    fig8.add_hline(
        y=yref, line_dash=dash, line_color="#B4B2A9", line_width=1.2,
        annotation_text=label_txt,
        annotation_position="right",
        annotation_font=dict(size=10, color="#888780"),
    )

fig8.update_layout(
    **LAYOUT_BASE,
    title=dict(
        text="Planet radius vs. model SNR — bubble size ∝ transit depth",
        x=0.02, font=dict(size=17, color="#2C2C2A")),
    xaxis=dict(title="Model SNR", showgrid=True,
               gridcolor="#E8E6DF", gridwidth=0.5, zeroline=False),
    yaxis=dict(title="Planet radius (R⊕)", showgrid=True,
               gridcolor="#E8E6DF", gridwidth=0.5, zeroline=False),
    legend=dict(orientation="v", x=0.99, y=0.99, xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#D3D1C7", borderwidth=0.5),
    height=460,
)
save(fig8, "8_planet_radius_snr_bubble")

