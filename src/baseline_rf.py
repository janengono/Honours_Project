"""baseline_rf.py -- Random Forest on the handcrafted features.csv, same
subject-level split as the Transformer. A classical-model sanity check."""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from splits import subject_level_split, apply_subject_split

this_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(this_dir, "..", "data", "processed", "features.csv"))

id_cols = ["subject_id", "class_id", "subject_name", "sample_name"]
error_cols = [c for c in df.columns if c.endswith("_error")]   # text columns, not features
feature_cols = [c for c in df.columns if c not in id_cols + error_cols]

X = df[feature_cols].values.astype(float)
y = df["class_id"].values
sids = df["subject_id"].values

for s in np.unique(sids):
    m = sids == s
    mu, sd = np.nanmean(X[m], axis=0), np.nanstd(X[m], axis=0)
    sd[sd == 0] = 1.0
    X[m] = (X[m] - mu) / sd

print(f"Using {len(feature_cols)} numeric features; "
      f"{int(df[feature_cols].isna().any(axis=1).sum())} trials have some missing values "
      f"(kept; Random Forest handles NaN natively)")

train_s, val_s, _ = subject_level_split(sids, seed=42)
X_tr, y_tr, _ = apply_subject_split(X, y, sids, train_s)
X_va, y_va, _ = apply_subject_split(X, y, sids, val_s)



print(f"Train: {len(X_tr)}  Val: {len(X_va)}\n")

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
pred = clf.predict(X_va)

print(f"Random Forest 5-class: accuracy={accuracy_score(y_va, pred):.4f}, "
      f"F1-macro={f1_score(y_va, pred, average='macro'):.4f}\n")
print(classification_report(y_va, pred, target_names=["BL", "PA1", "PA2", "PA3", "PA4"]))

import numpy as np
accs, f1s = [], []
for seed in [42, 1, 7, 123, 2024]:
    tr, va, _ = subject_level_split(sids, seed=seed)
    Xa, ya, _ = apply_subject_split(X, y, sids, tr)
    Xb, yb, _ = apply_subject_split(X, y, sids, va)
    p = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xa, ya).predict(Xb)
    accs.append(accuracy_score(yb, p)); f1s.append(f1_score(yb, p, average="macro"))
    print(f"seed={seed:5d}  acc={accs[-1]:.4f}  f1={f1s[-1]:.4f}")
print(f"RF: acc {np.mean(accs):.4f} +/- {np.std(accs):.4f}, f1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")