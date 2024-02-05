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