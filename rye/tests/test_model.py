import pytest

def test_forward():
    import torch
    from rye.models import RyeModel
    from rye.layers import RyeElman
    input_size = 8
    hidden_size = 16
    num_channels = 32
    length = 20
    repeat = 40
    model = RyeModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_channels=num_channels,
        layer=RyeElman,
        length=length,
        repeat=repeat,
    )
    batch_size = 10
    invariant_input = torch.randn(batch_size, input_size)
    equivariant_input = torch.randn(batch_size, 3)
    invariant_hidden = torch.randn(batch_size, hidden_size)
    equivariant_hidden = torch.randn(batch_size, 3, num_channels)
    probability = torch.rand(batch_size, batch_size).softmax(dim=-1)
    invariant_hidden, equivariant_hidden = model(
        probability=probability,
        invariant_input=invariant_input,
        equivariant_input=equivariant_input,
        invariant_hidden=invariant_hidden,
        equivariant_hidden=equivariant_hidden,
    )
    assert invariant_hidden.shape == (repeat, length, batch_size, hidden_size)
    assert equivariant_hidden.shape == (repeat, length, batch_size, 3, num_channels)

def test_rotation_equivariance(equivariance_test_utils):
    import torch
    from rye.layers import RyeElman
    from rye.models import RyeModel
    translation, rotation, reflection = equivariance_test_utils
    input_size = 8
    hidden_size = 16
    num_channels = 32
    length = 10
    repeat = 20
    model = RyeModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_channels=num_channels,
        layer=RyeElman,
        length=length,
        repeat=repeat,
    )
    batch_size = 10
    invariant_input = torch.randn(batch_size, input_size)
    equivariant_input = torch.randn(batch_size, 3)
    invariant_hidden = torch.randn(batch_size, hidden_size)
    equivariant_hidden = torch.randn(batch_size, 3, num_channels)
    probability = torch.zeros(batch_size, batch_size)

    # make sure there is only one random walk 
    probability[torch.randperm(batch_size), torch.arange(batch_size)] = 1

    _invariant_hidden, _equivariant_hidden = model(
        probability,
        invariant_input,
        equivariant_input,
        invariant_hidden,
        equivariant_hidden,
    )

    equivariant_input_rotated = rotation(equivariant_input)
    equivariant_hidden_rotated = rotation(equivariant_hidden.swapaxes(-1, -2)).swapaxes(-1, -2)
    _invariant_hidden_rotated, _equivariant_hidden_rotated = model(
        probability,
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
