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
    walk = generate_walk(probability, length, repeat=2)
    assert walk.shape == (2, 10, 3)

# def test_batch_walk():
#     import torch
#     from rye.walk import generate_walk
#     probability = torch.tensor([
#         [0.5, 0.5, 0.0],
#         [0.0, 0.5, 0.5],
#         [0.5, 0.0, 0.5],
#     ]).unsqueeze(0).expand(10, 3, 3)
#     length = 10
#     walk = generate_walk(probability, length)
#     print(walk.shape)