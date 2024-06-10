import pytest

def test_generate_walk_single():
    import torch
    from rye.walk import generate_walk
    probability = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
    ])
    length = 10
    walk = generate_walk(probability, length)
    assert walk.shape == (11, 3)
    


def test_generate_walk_batch():
    import torch
    from rye.walk import generate_walk
    probability = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
    ])
    probability = probability.unsqueeze(0).expand(2, 3, 3)
    length = 10
    walk = generate_walk(probability, length)
    assert walk.shape == (2, 11, 3)
    