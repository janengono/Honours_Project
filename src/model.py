"""
model.py
--------
Implements Section 3.3 (Deep Learning Framework) of the proposal:

  1. Modality-specific encoding branches: ECG, EDA, EMG (Section 3.3.2)
  2. Multimodal self-attention fusion (Equation 3.5)
  3. Pain intensity classification head over 5 classes (Section 3.3.3)

Input shape: (batch, seq_len=10, n_features=20), matching sequences.npz
produced by build_sequences.py. Features are split by modality using the
feature_names array saved alongside X, so this doesn't hardcode column
indices -- if segmentation.py's channel order ever changes, this still
works correctly.
"""

import math
import torch
import torch.nn as nn


def get_modality_indices(feature_names):
    """
    Given the flat list of 20 feature names (e.g. 'ecg_mean', 'gsr_std',
    'emg_trapezius_max', ...), returns a dict mapping modality name to the
    list of column indices belonging to it.
    """
    groups = {"ecg": [], "eda": [], "emg": []}
    for i, name in enumerate(feature_names):
        if name.startswith("ecg_"):
            groups["ecg"].append(i)
        elif name.startswith("gsr_"):
            groups["eda"].append(i)
        elif name.startswith("emg_"):
            groups["emg"].append(i)
        else:
            raise ValueError(f"Unrecognized feature name '{name}' -- can't assign to a modality")
    return groups


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal position embedding, added to each modality's
    per-timestep representation before its encoder, so the model knows the
    temporal order of the 10 sub-windows."""

    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class ModalityEncoder(nn.Module):
    """
    One modality-specific branch (Section 3.3.2): projects that modality's
    raw per-window stats into d_model, adds positional encoding, then runs
    a small Transformer encoder to learn temporal dependencies WITHIN that
    modality before fusion.
    """

    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        return self.encoder(x)  # (batch, seq_len, d_model)


class MultimodalAttentionFusion(nn.Module):
    """
    Implements Equation 3.5 explicitly (rather than relying on a library's
    black-box attention module), so attention weights are directly
    available for the explainability analysis in Section 3.4.

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Query, Key, and Value are all derived from the concatenation of the
    three modality-specific sequences, letting the model learn -- per
    timestep, per pain trial -- how much each modality's representation
    should attend to every other modality's (and its own) representation.
    This directly implements the proposal's claim that "attention weights
    allow the model to dynamically determine which physiological signals
    contribute most strongly during different pain states."
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, total_seq_len, d_model)  -- concatenated modality sequences
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, n_heads, seq_len, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_out(context)

        # Residual connection + layer norm (standard Transformer practice,
        # stabilizes training -- not explicitly in the proposal's equation
        # but necessary for it to train reliably)
        out = self.layer_norm(x + self.dropout(out))

        return out, attn_weights  # attn_weights returned for Section 3.4 explainability


class PainTransformer(nn.Module):
    """
    Full model: three modality encoders -> concatenate along the sequence
    axis -> multimodal attention fusion -> mean pool -> classification head.

    n_classes=5 to match BioVid's BL, PA1, PA2, PA3, PA4 (Section 3.3.3).
    """

    def __init__(self, modality_dims, d_model=32, n_heads=4, n_layers=1,
                 fusion_heads=4, n_classes=5, dropout=0.1):
        """
        modality_dims: dict like {'ecg': 4, 'eda': 4, 'emg': 12}, i.e. how
        many raw features each modality contributes per timestep. Get this
        from get_modality_indices() + counting.
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())

        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dim, d_model, n_heads, n_layers, dropout)
            for name, dim in modality_dims.items()
        })

        # Learned embedding added per modality so the fusion layer can tell
        # which modality each token in the concatenated sequence came from
        self.modality_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            for name in modality_dims
        })

        self.fusion = MultimodalAttentionFusion(d_model, n_heads=fusion_heads, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, modality_inputs, return_attention=False):
        """
        modality_inputs: dict {'ecg': tensor(batch, seq_len, 4),
                                'eda': tensor(batch, seq_len, 4),
                                'emg': tensor(batch, seq_len, 12)}
        Returns logits of shape (batch, n_classes). CrossEntropyLoss applies
        softmax internally (Equation 3.7), so raw logits are returned here.
        """
        encoded_sequences = []
        for name in self.modality_names:
            encoded = self.encoders[name](modality_inputs[name])       # (batch, seq_len, d_model)
            encoded = encoded + self.modality_embeddings[name]          # tag with modality identity
            encoded_sequences.append(encoded)

        fused_input = torch.cat(encoded_sequences, dim=1)  # (batch, n_modalities * seq_len, d_model)
        fused_output, attn_weights = self.fusion(fused_input)

        pooled = fused_output.mean(dim=1)  # (batch, d_model)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attn_weights
        return logits
