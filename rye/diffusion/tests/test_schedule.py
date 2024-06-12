import torch

def test_cosine_schedule():
    from rye.diffusion.schedule import cosine
    shape = torch.Size([3, 5])
    t0 = cosine(torch.zeros(shape))
    t1 = cosine(torch.ones(shape))
    assert (t0 > 0.0).all()
    assert (t1 < 0.0).all()
    assert t0.shape == shape
    assert t1.shape == shape

