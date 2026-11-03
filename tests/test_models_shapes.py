import numpy as np
import pytest
import torch

from windfc.models.paper_nets import PAPER_PARAM_COUNTS, TORCH_EXTRA_BIAS, build_net, count_params
from windfc.train import train_net


@pytest.mark.parametrize("arch", ["ann", "cnn", "rnn", "lstm"])
def test_param_counts_match_paper(arch):
    # Paper protocol: 10 inputs reshaped to (10 steps, 1 channel).
    net = build_net(arch, seq_len=10, channels=1)
    assert count_params(net) == PAPER_PARAM_COUNTS[arch] + TORCH_EXTRA_BIAS[arch]


@pytest.mark.parametrize("arch", ["ann", "cnn", "rnn", "lstm"])
@pytest.mark.parametrize("shape", [(9, 1), (24, 11)])
def test_forward_shapes(arch, shape):
    net = build_net(arch, *shape)
    out = net(torch.randn(5, *shape))
    assert out.shape == (5,)


def test_training_reduces_loss():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 12, 3)).astype(np.float32)
    y = (x[:, -1, 0] * 0.5 + 0.2).astype(np.float32)
    net = build_net("lstm", 12, 3)
    net, hist = train_net(net, x, y, x[:64], y[:64], epochs=15, batch_size=32, lr=3e-3, patience=50,
                          device="cpu", verbose=False)
    assert hist["train_loss"][-1] < hist["train_loss"][0] * 0.5
