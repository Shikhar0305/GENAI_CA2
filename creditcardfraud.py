# %% [markdown]
# # Credit Card Fraud Detection: Random Forest (train, save, load & test `model.joblib`)
#
# Steps: load data → remove duplicates → train/validation/test split → train Random Forest →
# tune decision threshold on validation → evaluate on test → save model + parameters → reload and verify.
#
# Only ~0.17% of transactions are fraud, so we judge the model by **PR-AUC, precision, recall and F1**
# for the fraud class, not accuracy.

# %% [markdown]
# ### 1. Load and Explore the Dataset

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, average_precision_score,
                             roc_auc_score, precision_recall_curve, precision_score,
                             recall_score, f1_score)

RANDOM_STATE = 42
DATA_PATH = os.getenv('DATA_PATH', 'creditcard.csv')

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nClass distribution:")
print(df['Class'].value_counts())
print(f"Fraud rate: {df['Class'].mean() * 100:.3f}%")
print(df.head())

# %% [markdown]
# ### 2. Clean Data and Split into Train / Validation / Test Sets
# - Duplicate rows are removed so the same transaction can't appear in both the training and test sets.
# - No scaling is needed: Random Forests don't depend on feature scale, so raw `Amount` is used.
#   This also means `model.joblib` works directly on raw transaction data.
# - Split: 60% train, 20% validation (threshold tuning), 20% test (final evaluation only).

# %%
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

FEATURES = [c for c in df.columns if c not in ('Class', 'Time')]
X = df[FEATURES]
y = df['Class']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)

for name, part in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
    print(f"{name:10s}: {len(part):6d} rows, {part.sum():3d} frauds ({part.mean() * 100:.3f}%)")

# %% [markdown]
# ### 3. Train the Random Forest Model

# %%
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)
print("Model trained.")

# %% [markdown]
# ### 4. Tune the Decision Threshold on the Validation Set
# The default threshold of 0.5 is rarely best for imbalanced data. We choose the probability
# threshold that gives the highest F1 score for the fraud class on the validation set.

# %%
p_val = model.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, p_val)
f1_vals = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_idx = int(np.argmax(f1_vals))
THRESHOLD = float(thr[best_idx])

val_metrics = {
    'PR_AUC': round(average_precision_score(y_val, p_val), 4),
    'ROC_AUC': round(roc_auc_score(y_val, p_val), 4),
}
print(f"Best threshold: {THRESHOLD:.3f}")
print(f"Validation at this threshold -> precision={prec[best_idx]:.3f}, "
      f"recall={rec[best_idx]:.3f}, F1={f1_vals[best_idx]:.3f}")
print("Validation metrics:", val_metrics)

plt.figure(figsize=(6, 4))
plt.plot(rec, prec)
plt.scatter(rec[best_idx], prec[best_idx], color='red', zorder=3,
            label=f'Chosen threshold = {THRESHOLD:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ### 5. Evaluate on the Held-out Test Set

# %%
p_test = model.predict_proba(X_test)[:, 1]
y_pred = (p_test >= THRESHOLD).astype(int)

test_metrics = {
    'PR_AUC': round(average_precision_score(y_test, p_test), 4),
    'ROC_AUC': round(roc_auc_score(y_test, p_test), 4),
    'precision_fraud': round(precision_score(y_test, y_pred), 4),
    'recall_fraud': round(recall_score(y_test, y_pred), 4),
    'f1_fraud': round(f1_score(y_test, y_pred), 4),
}
print("Test metrics:")
print(json.dumps(test_metrics, indent=2))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraud (1)'], digits=4))

# %% [markdown]
# ### 6. Save the Model and Its Parameters to `model.joblib`

# %%
model_payload = {
    'model': model,
    'features': FEATURES,
    'threshold': THRESHOLD,
    'hyperparameters': model.get_params(),
    'metrics': {'validation': val_metrics, 'test': test_metrics},
    'sklearn_version': sklearn.__version__,
}
joblib.dump(model_payload, 'model.joblib', compress=3)
print(f"Saved 'model.joblib' ({os.path.getsize('model.joblib') / 1024 / 1024:.2f} MB)")

# Download to your computer when running in Colab
try:
    from google.colab import files
    files.download('model.joblib')
except ImportError:
    pass

# %% [markdown]
# ### 7. Load the Model and Verify Parameters / Performance

# %%
loaded_payload = joblib.load('model.joblib')
loaded_model = loaded_payload['model']
loaded_features = loaded_payload['features']
loaded_threshold = loaded_payload['threshold']
saved_params = loaded_payload['hyperparameters']

print("Saved with scikit-learn", loaded_payload['sklearn_version'], "| running", sklearn.__version__)
print("Number of features:", len(loaded_features))
print(f"Threshold: {loaded_threshold:.3f}")
print("Saved Hyperparameters:")
for param in ['n_estimators', 'min_samples_leaf', 'class_weight', 'max_depth', 'random_state']:
    print(f"  {param}: {saved_params[param]}")

# Test 1: saved parameters match the trained model
assert saved_params == model.get_params(), "Hyperparameters differ!"
assert loaded_features == FEATURES and loaded_threshold == THRESHOLD, "Features/threshold differ!"
print("\n✅ Test 1 passed: saved hyperparameters, features and threshold match the trained model.")

# Test 2: loaded model gives exactly the same predictions as the original
p_loaded = loaded_model.predict_proba(X_test[loaded_features])[:, 1]
y_loaded = (p_loaded >= loaded_threshold).astype(int)
assert np.allclose(p_loaded, p_test) and (y_loaded == y_pred).all(), "Predictions differ!"
print("✅ Test 2 passed: loaded model reproduces the original predictions exactly.")

# Test 3: metrics recomputed from the loaded model match the saved metrics
recomputed = {
    'PR_AUC': round(average_precision_score(y_test, p_loaded), 4),
    'ROC_AUC': round(roc_auc_score(y_test, p_loaded), 4),
    'precision_fraud': round(precision_score(y_test, y_loaded), 4),
    'recall_fraud': round(recall_score(y_test, y_loaded), 4),
    'f1_fraud': round(f1_score(y_test, y_loaded), 4),
}
assert recomputed == loaded_payload['metrics']['test'], "Metrics differ!"
print("✅ Test 3 passed: metrics from the loaded model match the saved metrics:", recomputed)

print("\nClassification Report (loaded model):")
print(classification_report(y_test, y_loaded, target_names=['Legitimate (0)', 'Fraud (1)'], digits=4))

# %% [markdown]
# ### 8. Predict on New Transactions with the Loaded Model

# %%
def predict_fraud(transactions, payload=loaded_payload):
    """Return fraud probability and 0/1 prediction for raw transaction rows."""
    proba = payload['model'].predict_proba(transactions[payload['features']])[:, 1]
    return pd.DataFrame({'fraud_probability': proba.round(4),
                         'prediction': (proba >= payload['threshold']).astype(int)},
                        index=transactions.index)

sample = pd.concat([X_test[y_test == 1].head(3), X_test[y_test == 0].head(3)])
result = predict_fraud(sample)
result['actual'] = y_test.loc[sample.index].values
print(result)

# %% [markdown]
# ### 9. Visualize the Confusion Matrix

# %%
cm = confusion_matrix(y_test, y_loaded)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate (0)', 'Fraud (1)'],
            yticklabels=['Legitimate (0)', 'Fraud (1)'])
plt.title(f'Confusion Matrix - Random Forest (threshold = {loaded_threshold:.3f})')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.show()