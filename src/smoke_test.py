"""
smoke_test.py
-------------
Sanity-checks the Transformer + training pipeline using SYNTHETIC data of
the same shape as the real sequences.npz, so bugs in model.py/train.py/
pain_dataset.py surface immediately without waiting through a real training
run. Run this FIRST after installing requirements, before touching real data.

    python smoke_test.py

Expected: prints shapes at each stage, trains for 2 tiny epochs, and exits
with no errors. Loss won't mean anything (data is random) -- this only
checks that tensors flow through the architecture correctly.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PainTransformer, get_modality_indices
from pain_dataset import PainSequenceDataset, modality_feature_dims, collate_fn

print("=== Synthetic data ===")
n_trials, seq_len, n_features = 200, 10, 20
feature_names = (
    [f"ecg_{s}" for s in ["mean", "std", "min", "max"]]
    + [f"gsr_{s}" for s in ["mean", "std", "min", "max"]]
    + [f"emg_trapezius_{s}" for s in ["mean", "std", "min", "max"]]
    + [f"emg_corrugator_{s}" for s in ["mean", "std", "min", "max"]]
    + [f"emg_zygomaticus_{s}" for s in ["mean", "std", "min", "max"]]
)
assert len(feature_names) == n_features

rng = np.random.default_rng(0)
X = rng.normal(size=(n_trials, seq_len, n_features)).astype(np.float32)
y = rng.integers(0, 5, size=n_trials)
subject_ids = rng.integers(1, 11, size=n_trials)  # 10 fake subjects

modality_indices = get_modality_indices(feature_names)
mod_dims = modality_feature_dims(modality_indices)
print(f"Modality indices: {modality_indices}")
print(f"Modality dims: {mod_dims}")

dataset = PainSequenceDataset(X, y, subject_ids, feature_names, modality_indices)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

modality_tensors, labels = next(iter(loader))
print("\n=== Batch shapes ===")
for name, t in modality_tensors.items():
    print(f"  {name}: {t.shape}")
print(f"  labels: {labels.shape}")

print("\n=== Model forward pass ===")
model = PainTransformer(modality_dims=mod_dims, d_model=32, n_heads=4, n_layers=1, fusion_heads=4, n_classes=5)
logits = model(modality_tensors)
print(f"  logits shape: {logits.shape}  (expected: [16, 5])")
assert logits.shape == (16, 5), "Logits shape mismatch!"

logits, attn = model(modality_tensors, return_attention=True)
print(f"  attention weights shape: {attn.shape}  (expected: [16, 4, 30, 30])")

print("\n=== Tiny training loop (2 epochs, synthetic data) ===")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2):
    model.train()
    total_loss = 0.0
    for modality_tensors, labels in loader:
        optimizer.zero_grad()
        logits = model(modality_tensors)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    print(f"  Epoch {epoch + 1}: loss = {total_loss / len(dataset):.4f}")

print("\nSMOKE TEST PASSED -- architecture, dataset, and training loop all run without errors.")
print("Loss values are meaningless here (random data) -- this only confirms nothing crashes.")
