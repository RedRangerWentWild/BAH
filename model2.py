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