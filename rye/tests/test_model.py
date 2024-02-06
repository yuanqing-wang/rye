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