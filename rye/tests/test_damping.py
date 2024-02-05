import pytest

def test_damping_equivariance(equivariance_test_utils):
    import torch
    from rye.layers import Damping
    translation, rotation, reflection = equivariance_test_utils
    batch_size = 10
    num_channels = 32
    hidden_features = 16
    layer = Damping(hidden_features, num_channels)
    x = torch.randn(batch_size, 3, num_channels)
    h = torch.randn(batch_size, hidden_features)
    y = layer(x, h)
    y_rotated = layer(rotation(x.swapaxes(-1, -2)).swapaxes(-1, -2), h)

    assert torch.allclose(
        rotation(y.swapaxes(-1, -2)).swapaxes(-1, -2),
        y_rotated,
        rtol=1e-3,
    )