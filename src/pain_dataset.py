"""
pain_dataset.py
----------------
Loads data/processed/sequences.npz and wraps it as a PyTorch Dataset that
returns per-modality tensors (matching what model.PainTransformer expects),
plus utilities to build DataLoaders for a given subject split.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from model import get_modality_indices


class PainSequenceDataset(Dataset):
    def __init__(self, X, y, subject_ids, feature_names, modality_indices):
        """
        X: np.ndarray (n_trials, seq_len, n_features)
        y: np.ndarray (n_trials,)
        subject_ids: np.ndarray (n_trials,)
        modality_indices: dict from get_modality_indices()
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.subject_ids = subject_ids
        self.modality_indices = modality_indices

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X[idx]  # (seq_len, n_features)
        modality_tensors = {
            name: torch.from_numpy(row[:, idx_list])
            for name, idx_list in self.modality_indices.items()
        }
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return modality_tensors, label


def load_sequence_data(config_path=None):
    """Loads sequences.npz relative to this file's project root."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    seq_path = os.path.join(this_dir, "..", "data", "processed", "sequences.npz")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(
            f"{seq_path} not found -- run build_sequences.py first."
        )
    data = np.load(seq_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    subject_ids = data["subject_ids"]
    feature_names = list(data["feature_names"])
    return X, y, subject_ids, feature_names


def modality_feature_dims(modality_indices):
    """e.g. {'ecg': 4, 'eda': 4, 'emg': 12}, needed to construct PainTransformer."""
    return {name: len(idxs) for name, idxs in modality_indices.items()}


def collate_fn(batch):
    """Stacks a list of (modality_dict, label) into batched tensors."""
    modality_names = batch[0][0].keys()
    batched_modalities = {
        name: torch.stack([item[0][name] for item in batch]) for name in modality_names
    }
    labels = torch.stack([item[1] for item in batch])
    return batched_modalities, labels


if __name__ == "__main__":
    # Smoke test
    X, y, subject_ids, feature_names = load_sequence_data()
    modality_indices = get_modality_indices(feature_names)
    print("Modality indices:", modality_indices)
    print("Modality dims:", modality_feature_dims(modality_indices))

    dataset = PainSequenceDataset(X, y, subject_ids, feature_names, modality_indices)
    print(f"Dataset size: {len(dataset)}")

    modality_tensors, label = dataset[0]
    for name, t in modality_tensors.items():
        print(f"  {name}: shape {t.shape}")
    print(f"  label: {label.item()}")
