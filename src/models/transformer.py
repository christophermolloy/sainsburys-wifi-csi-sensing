"""
models/transformer.py
---------------------
Transformer encoder for WiFi CSI activity classification.

Input tensor shape: ``(B, T, S)``
  - B = batch size
  - T = time steps (sequence length), e.g. 250
  - S = number of subcarriers (raw feature dim), e.g. 90

Architecture overview
---------------------
1. **Linear projection** – maps each time-step's S-dimensional subcarrier
   vector to a ``d_model``-dimensional token embedding.  This decouples the
   model width from the hardware-specific subcarrier count, making the
   Transformer reusable across different 802.11 configurations.

2. **Sinusoidal positional encoding** – injects absolute temporal order using
   the fixed sin/cos scheme from "Attention Is All You Need".  Sinusoidal
   encodings are preferred over learnable ones here because CSI windows may
   be of variable length and the encoding generalises without retraining.

3. **TransformerEncoder** – ``num_encoder_layers`` layers of multi-head
   self-attention + position-wise feed-forward, with pre-norm (LayerNorm
   before each sublayer) for training stability.

4. **Temporal mean pooling** – averages across the time axis to obtain a
   fixed-size sequence representation, which is more robust than taking the
   CLS token when motion is distributed throughout the window.

5. **Classification head** – LayerNorm → Dropout → Linear.

Why Transformers for CSI?
--------------------------
Self-attention has global receptive field (every time step attends to every
other), making it ideal for capturing long-range temporal dependencies such
as the periodicity of walking (≈ 0.5–1 s cycle) or the slow drift during
sitting.  The multi-head mechanism also implicitly learns which subcarrier
groups are informative for a given activity.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Adds position information to token embeddings without any learnable
    parameters:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The buffer is pre-computed for ``max_len`` positions and sliced at
    runtime, so variable-length sequences up to ``max_len`` are supported
    with zero overhead.

    Parameters
    ----------
    d_model : int
        Embedding / model dimension.
    dropout : float
        Dropout applied to the summed embedding + positional encoding.
    max_len : int
        Maximum sequence length to pre-compute.  Default: 5000 (covers
        CSI windows up to 20 s at 250 Hz).
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build (max_len, d_model) encoding table.
        pe = torch.zeros(max_len, d_model)                   # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        # Division terms for each dimension pair.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )                                                    # (D/2,)
        pe[:, 0::2] = torch.sin(position * div_term)        # even dims
        pe[:, 1::2] = torch.cos(position * div_term)        # odd dims
        # Store as (1, max_len, d_model) for broadcasting over the batch.
        pe = pe.unsqueeze(0)
        # Register as buffer so it moves with .to(device) but is not a param.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to token embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Token embeddings, shape ``(B, T, d_model)``.

        Returns
        -------
        torch.Tensor
            Positionally encoded embeddings, same shape.
        """
        # self.pe: (1, max_len, d_model) → slice to (1, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CSITransformer(nn.Module):
    """Transformer encoder model for CSI-based activity classification.

    Parameters
    ----------
    num_classes : int
        Number of output activity classes.
    input_size : int, optional
        Raw feature dimension per time step (number of subcarriers).
        Default: 90.
    d_model : int, optional
        Internal embedding dimension.  Must be divisible by ``nhead``.
        Default: 64.
    nhead : int, optional
        Number of self-attention heads.  Default: 4.
    num_encoder_layers : int, optional
        Number of TransformerEncoderLayer blocks.  Default: 2.
    dim_feedforward : int, optional
        Hidden dimension of the position-wise FFN inside each encoder layer.
        Default: 256.
    dropout : float, optional
        Dropout rate used throughout (positional encoding, attention,
        FFN, head).  Default: 0.1.
    max_len : int, optional
        Maximum supported sequence length for positional encoding.
        Default: 5000.

    Raises
    ------
    ValueError
        If ``d_model`` is not divisible by ``nhead``.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int = 90,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        # ---- Input projection ----
        # Maps raw subcarrier features → d_model tokens.
        self.input_projection = nn.Linear(input_size, d_model)

        # ---- Positional encoding ----
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )

        # ---- Transformer encoder ----
        # norm_first=True implements pre-LN (LayerNorm before each sublayer),
        # which yields more stable gradients than post-LN in small-data regimes.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",   # GELU is standard in modern Transformers
            batch_first=True,    # expects (B, T, d_model)
            norm_first=True,     # pre-LN for training stability
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm,
        )

        # ---- Classification head ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialisation for projection and classification layers."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: "Optional[torch.Tensor]" = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw CSI features, shape ``(B, T, S)`` where S == ``input_size``.
        src_key_padding_mask : torch.Tensor, optional
            Boolean mask of shape ``(B, T)``.  ``True`` at positions that
            should be *ignored* (e.g. padding in variable-length batches).
            Default: None (no masking).

        Returns
        -------
        torch.Tensor
            Class logits, shape ``(B, num_classes)``.

        Notes
        -----
        For fixed-length windows (the common case in CSI), omit
        ``src_key_padding_mask``.  Pass it when you pack variable-length
        clips into a single batch for efficiency.
        """
        # 1. Project subcarrier features to d_model tokens.
        #    (B, T, S) → (B, T, d_model)
        tokens = self.input_projection(x)

        # 2. Add sinusoidal positional encoding and apply dropout.
        #    (B, T, d_model) → (B, T, d_model)
        tokens = self.pos_encoding(tokens)

        # 3. Run through the Transformer encoder stack.
        #    (B, T, d_model) → (B, T, d_model)
        encoded = self.encoder(
            tokens, src_key_padding_mask=src_key_padding_mask
        )

        # 4. Temporal mean pooling → (B, d_model).
        #    If a padding mask is provided, exclude masked positions from the mean.
        if src_key_padding_mask is not None:
            # mask: True = padding → weight 0; invert and normalise.
            mask_float = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            pooled = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = encoded.mean(dim=1)  # (B, d_model)

        # 5. Classification head.
        return self.classifier(pooled)  # (B, num_classes)
