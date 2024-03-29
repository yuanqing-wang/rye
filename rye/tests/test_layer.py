import pytest

def test_forward():
    import torch
    from rye.layers import RyeElman
    input_size = 8
    hidden_size = 16
    num_channels = 32
    layer = RyeElman(
        input_size=input_size,
        hidden_size=hidden_size,
        num_channels=num_channels,
    )
    batch_size = 10
    invariant_input = torch.randn(batch_size, input_size)
    equivariant_input = torch.randn(batch_size, 3)
    invariant_hidden = torch.randn(batch_size, hidden_size)
    equivariant_hidden = torch.randn(batch_size, 3, num_channels)
    invariant_hidden, equivariant_hidden = layer(
        invariant_input,
        equivariant_input,
        invariant_hidden,
        equivariant_hidden,
    )
    assert invariant_hidden.shape == (batch_size, hidden_size)
    assert equivariant_hidden.shape == (batch_size, 3, num_channels)

def test_repeat():
    import torch
    from rye.layers import RyeElman
    input_size = 8
    hidden_size = 16
    num_channels = 32
    layer = RyeElman(
        input_size=input_size,
        hidden_size=hidden_size,
        num_channels=num_channels,
    )
    repeat = 20
    batch_size = 10
    invariant_input = torch.randn(20, batch_size, input_size)
    equivariant_input = torch.randn(20, batch_size, 3)
    invariant_hidden = torch.randn(20, batch_size, hidden_size)
    equivariant_hidden = torch.randn(20, batch_size, 3, num_channels)
    invariant_hidden, equivariant_hidden = layer(
        invariant_input,
        equivariant_input,
        invariant_hidden,
        equivariant_hidden,
    )
    assert invariant_hidden.shape == (repeat, batch_size, hidden_size)
    assert equivariant_hidden.shape == (repeat, batch_size, 3, num_channels)

def test_equivariance(equivariance_test_utils):
    import torch
    from rye.layers import RyeElman
    translation, rotation, reflection = equivariance_test_utils
    input_size = 8
    hidden_size = 16
    num_channels = 32
    layer = RyeElman(
        input_size=input_size,
        hidden_size=hidden_size,
        num_channels=num_channels,
    )
    batch_size = 10
    invariant_input = torch.randn(batch_size, input_size)
    equivariant_input = torch.randn(batch_size, 3)
    invariant_hidden = torch.randn(batch_size, hidden_size)
    equivariant_hidden = torch.randn(batch_size, 3, num_channels)
    _invariant_hidden, _equivariant_hidden = layer(
        invariant_input,
        equivariant_input,
        invariant_hidden,
        equivariant_hidden,
    )

    equivariant_input_rotated = rotation(equivariant_input)
    equivariant_hidden_rotated = rotation(equivariant_hidden.swapaxes(-1, -2)).swapaxes(-1, -2)
    _invariant_hidden_rotated, _equivariant_hidden_rotated = layer(
        invariant_input,
        equivariant_input_rotated,
        invariant_hidden,
        equivariant_hidden_rotated,
    )

    assert torch.allclose(
        _invariant_hidden,
        _invariant_hidden_rotated,
        atol=1e-2,
        rtol=1e-2,
    )

    assert torch.allclose(
        _equivariant_hidden_rotated,
        rotation(_equivariant_hidden.swapaxes(-1, -2)).swapaxes(-1, -2),
        atol=1e-2,
        rtol=1e-2,
    )