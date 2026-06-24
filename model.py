import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/kepler_labelled.csv')

X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model for later use
import pickle
with open('data/model.pkl', 'wb') as f:
    pickle.dump((model, le), f)
print("Model saved to data/model.pkl")

# Fix confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Fix feature importance plot
fig, ax = plt.subplots(figsize=(8, 6))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Importance')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plots saved!")

# Add this to model.py after saving the first model

print("\n--- Model 2: Physics-only (no flags) ---")
physics_features = [
    'koi_period', 'koi_duration', 'koi_depth',
    'koi_model_snr', 'koi_prad', 'koi_score',
    'koi_impact', 'koi_steff', 'koi_slogg', 'koi_srad',
    'koi_period_err1', 'koi_depth_err1', 'koi_duration_err1'
]

X2 = df[physics_features]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model2 = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42
)
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)
print(classification_report(y2_test, y2_pred, target_names=le.classes_))

# Feature importance for physics model
fig, ax = plt.subplots(figsize=(8, 6))
importances2 = pd.Series(model2.feature_importances_, index=X2.columns)
importances2.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Importance - Physics Only Model')
plt.tight_layout()
plt.savefig('feature_importance_physics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_importance_physics.png")

with open('data/model2_physics.pkl', 'wb') as f:
    pickle.dump((model2, le), f)