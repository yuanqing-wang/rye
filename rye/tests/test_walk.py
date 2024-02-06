import pytest

def test_generate_walk():
    import torch
    from rye.walk import generate_walk
    probability = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
    ])
    length = 10
    walk = generate_walk(probability, length)
    assert walk.shape == (length, 3)
    assert (walk >= 0).all()
    assert (walk < 3).all()