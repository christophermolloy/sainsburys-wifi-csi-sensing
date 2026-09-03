"""
models/lstm.py
--------------
Bidirectional LSTM (with optional self-attention) for WiFi CSI time-series
activity classification.

Input tensor shape: ``(B, T, S)``
  - B = batch size
  - T = time steps (e.g. 250)
  - S = number of subcarriers (e.g. 90) — treated as the input feature dim

Architecture overview
---------------------
1. **BiLSTM encoder** – processes the subcarrier-indexed feature vector at
   each time step.  Bidirectionality lets the model use future context, which
   is valid for offline / sliding-window inference.

2. **Temporal attention** (optional) – learns a scalar importance weight for
   every time step and produces a single weighted-sum context vector.  This is
   particularly useful for CSI because motion artefacts often occupy only a
   short sub-window of the full capture.

3. **Classification head** – LayerNorm → Dropout → Linear.

Why LSTM for CSI?
-----------------
CSI amplitude fluctuates on timescales of tens to hundreds of milliseconds.
The LSTM's gated memory is well-suited to capture long-range temporal
dependencies (e.g. the onset and decay of a walking cycle) that plain
convolutional models can miss when the field of view is limited.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) self-attention over the time axis.

    Given LSTM hidden states ``H`` of shape ``(B, T, H)``, computes a
    context vector ``c`` of shape ``(B, H)`` as a weighted sum over time:

        e_t = tanh(W * h_t + b)          # energy at time t
        α   = softmax(e)                  # attention weights (B, T)
        c   = Σ_t α_t * h_t              # context vector (B, H)

    The single-layer energy function is light-weight and interpretable:
    plotting ``α`` reveals *which time steps* drove the classification
    decision, which is valuable for debugging CSI pipelines.

    Parameters
    ----------
    hidden_size : int
        Dimension of LSTM hidden states fed in (after bidirectionality
        doubling has already been applied by the caller).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        # Project hidden states to scalar energies.
        self.energy = nn.Linear(hidden_size, 1, bias=True)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector and attention weights.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(B, T, H)``.

        Returns
        -------
        context : torch.Tensor
            Weighted sum of hidden states, shape ``(B, H)``.
        weights : torch.Tensor
            Normalised attention weights, shape ``(B, T)``.
        """
        # (B, T, 1) → (B, T)
        energies = self.energy(torch.tanh(hidden_states)).squeeze(-1)
        weights = F.softmax(energies, dim=1)                # (B, T)
        # Weighted sum: (B, T, 1) * (B, T, H) → (B, H)
        context = (weights.unsqueeze(-1) * hidden_states).sum(dim=1)
        return context, weights


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CSILSTM(nn.Module):
    """Bidirectional LSTM with optional temporal attention for CSI classification.

    Parameters
    ----------
    num_classes : int
        Number of output activity classes.
    input_size : int, optional
        Number of input features per time step (i.e. number of subcarriers).
        Default: 90.
    hidden_size : int, optional
        Number of LSTM units per direction.  Default: 128.
    num_layers : int, optional
        Number of stacked LSTM layers.  Default: 2.
    dropout : float, optional
        Dropout probability.  Applied between LSTM layers (when
        ``num_layers > 1``) and before the classification head.  Default: 0.5.
    bidirectional : bool, optional
        Whether to use a bidirectional LSTM.  Default: True.
    use_attention : bool, optional
        Whether to apply temporal self-attention over LSTM outputs.  When
        False the model uses the concatenated final hidden states instead.
        Default: True.

    Notes
    -----
    When ``bidirectional=True`` the effective hidden dimension of each LSTM
    output is ``2 * hidden_size``.  The classification head is sized
    accordingly.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int = 90,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        # The effective output feature dimension of the LSTM.
        self.lstm_out_dim = hidden_size * self.num_directions

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        if use_attention:
            self.attention = TemporalAttention(self.lstm_out_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.lstm_out_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.lstm_out_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialisation for LSTM weights; Xavier for FC."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                # Zero bias with forget-gate bias = 1 to aid gradient flow
                # at the start of training.
                param.data.fill_(0)
                n = param.size(0)
                # forget gate occupies positions [n/4 : n/2]
                param.data[n // 4 : n // 2].fill_(1.0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # noqa: E501 — returns (logits,) or (logits, attn_weights) depending on use_attention
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, S)`` where S == ``input_size``.

        Returns
        -------
        logits : torch.Tensor
            Class logits, shape ``(B, num_classes)``.

        Notes
        -----
        During *inference* you may want attention weights for interpretability.
        Call ``model.get_attention_weights(x)`` instead of ``forward`` to
        retrieve both logits and weights.
        """
        # lstm_out: (B, T, num_directions * hidden_size)
        lstm_out, (h_n, _) = self.lstm(x)

        if self.use_attention:
            context, _ = self.attention(lstm_out)  # (B, lstm_out_dim)
        else:
            # Concatenate the final hidden states from all directions.
            # h_n shape: (num_layers * num_directions, B, hidden_size)
            # Take only the last layer's hidden states.
            if self.bidirectional:
                # Forward direction: h_n[-2], backward: h_n[-1]
                context = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, lstm_out_dim)
            else:
                context = h_n[-1]  # (B, hidden_size)

        return self.classifier(context)

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits *and* attention weights for interpretability.

        Only callable when ``use_attention=True``.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, S)``.

        Returns
        -------
        logits : torch.Tensor
            ``(B, num_classes)``.
        weights : torch.Tensor
            Attention weight distribution over time, ``(B, T)``.

        Raises
        ------
        RuntimeError
            If the model was built without attention (``use_attention=False``).
        """
        if not self.use_attention:
            raise RuntimeError(
                "This model was instantiated with use_attention=False. "
                "Re-initialise with use_attention=True to use this method."
            )
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits, weights
