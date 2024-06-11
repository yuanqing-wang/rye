import torch
from typing import Callable
from . import schedule

class Diffusion(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Module,
            schedule: Callable = schedule.cosine,
    ):
        super().__init__()
        self.model = model
        self.schedule = schedule

    def loss(
            self,
            x: torch.Tensor,
    ):
        # sample time
        t = torch.empty(
            x.size(0),
        ).uniform_(0, 1)

        # sample noise
        epsilon = torch.randn_like(x)
        
        # calculate the log SNR
        log_snr = self.schedule(t)

        # calculate the noised input
        alpha, sigma = log_snr.sigmoid().sqrt(), (-log_snr).sigmoid().sqrt()
        x_noised =  x * alpha + epsilon * sigma

        # calculate the loss
        epsilon_hat = self.model(x_noised)
        loss = (epsilon_hat - epsilon).pow(2).mean()
        return loss




