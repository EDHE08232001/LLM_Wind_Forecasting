"""PyTorch versions of the four networks in Karaman (2023), Tables 1-4.

All nets take ``x`` of shape (batch, seq_len, channels) and return (batch,).

* Protocol A (paper): the feature vector is treated as a 1-D sequence
  (seq_len = n_features, channels = 1), exactly like the Keras reshape implied
  by the paper's parameter counts (Conv1D output (8, 32) from 10 inputs).
* Protocol B (ours): a real look-back window (seq_len = lookback hours,
  channels = number of input variables).

Parameter counts with 10 inputs (paper protocol): ANN 2,817 and CNN 10,497
match the paper exactly. PyTorch recurrent layers carry two bias vectors
(``b_ih`` and ``b_hh``) where Keras has one, so RNN = 5,313 + 32 = 5,345 and
LSTM = 31,233 + 256 = 31,489.
"""

from __future__ import annotations

import torch
from torch import nn


def _head(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )


class ANN(nn.Module):
    """Dense 64 -> 32 -> 1 on the flattened input."""

    def __init__(self, seq_len: int, channels: int):
        super().__init__()
        self.net = _head(seq_len * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).squeeze(-1)


class CNN(nn.Module):
    """Conv1D(32, k=3) -> MaxPool(2) -> Flatten -> Dense 64 -> 32 -> 1."""

    def __init__(self, seq_len: int, channels: int, filters: int = 32, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(channels, filters, kernel_size=kernel)
        self.pool = nn.MaxPool1d(2)
        conv_len = seq_len - kernel + 1
        if conv_len < 2:
            raise ValueError(f"CNN needs seq_len >= {kernel + 1}, got {seq_len}")
        self.head = _head(filters * (conv_len // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.conv(x.transpose(1, 2)))  # (B, C, L) -> (B, F, L')
        z = self.pool(z)
        return self.head(z.flatten(1)).squeeze(-1)


class RNN(nn.Module):
    """SimpleRNN(32, ReLU) -> Dense 64 -> 32 -> 1."""

    def __init__(self, seq_len: int, channels: int, hidden: int = 32):
        super().__init__()
        self.rnn = nn.RNN(channels, hidden, batch_first=True, nonlinearity="relu")
        self.head = _head(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(x)
        return self.head(h[-1]).squeeze(-1)


class LSTMNet(nn.Module):
    """Conv1D(32, k=3) -> LSTM(64) -> Dense 64 -> 32 -> 1."""

    def __init__(self, seq_len: int, channels: int, filters: int = 32, kernel: int = 3, hidden: int = 64):
        super().__init__()
        if seq_len < kernel:
            raise ValueError(f"LSTM net needs seq_len >= {kernel}, got {seq_len}")
        self.conv = nn.Conv1d(channels, filters, kernel_size=kernel)
        self.lstm = nn.LSTM(filters, hidden, batch_first=True)
        self.head = _head(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)  # (B, L', F)
        _, (h, _) = self.lstm(z)
        return self.head(h[-1]).squeeze(-1)


NETS = {"ann": ANN, "cnn": CNN, "rnn": RNN, "lstm": LSTMNet}


def build_net(name: str, seq_len: int, channels: int) -> nn.Module:
    try:
        cls = NETS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown net {name!r}; choose from {list(NETS)}") from exc
    return cls(seq_len, channels)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Paper (Keras) totals for reference / tests.
PAPER_PARAM_COUNTS = {"ann": 2817, "cnn": 10497, "rnn": 5313, "lstm": 31233}
# Extra PyTorch recurrent bias (b_hh) relative to Keras.
TORCH_EXTRA_BIAS = {"ann": 0, "cnn": 0, "rnn": 32, "lstm": 4 * 64}
