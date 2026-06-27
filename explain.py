import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
with open('data/model.pkl', 'rb') as f:
    model, le = pickle.load(f)

df = pd.read_csv('data/kepler_labelled.csv')
X = df.drop('label', axis=1)
y = df['label']

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot 1: Summary plot
plt.figure()
shap.summary_plot(shap_values, X, class_names=le.classes_, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_summary.png")

# Plot 2: Waterfall for one transit example
transit_examples = df[df['label'] == 'transit'].head(1)
X_example = transit_examples.drop('label', axis=1)
transit_idx = list(le.classes_).index('transit')

explainer2 = shap.TreeExplainer(model)
shap_single = explainer2(X_example)

plt.figure()
shap.waterfall_plot(shap_single[0, :, transit_idx], show=False)
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_waterfall.png")