import pytest

def test_dot_product_rotation_equivariance(equivariance_test_utils):
    import torch
    from rye.layers import DotProductProjection
    translation, rotation, reflection = equivariance_test_utils
    batch_size = 10
    in_features = 4
    out_features = 8
    layer = DotProductProjection(in_features, out_features)
    x = torch.randn(batch_size, 3, in_features)
    y = layer(x)
    y_rotated = layer(rotation(x.swapaxes(-1, -2)).swapaxes(-1, -2))
    assert torch.allclose(y_rotated, y, atol=1e-3, rtol=1e-3)