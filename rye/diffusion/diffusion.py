import torch
from typing import Callable
from . import schedule

class Diffusion(torch.nn.Module):
    def __init__(
            self,
            schedule: Callable = schedule.cosine,
            steps: int = 100,
    ):
        super().__init__()
        self.schedule = schedule
        self.steps = steps

    def loss(
            self,
            model: Callable,
            x: torch.Tensor,
    ):
        # sample time
        t = torch.empty(
            x.size(0),
            device=x.device,
        ).uniform_(0, 1)

        # sample noise
        epsilon = torch.randn_like(x)
        
        # calculate the log SNR
        log_snr = self.schedule(t)
        log_snr = log_snr.view(
            *log_snr.shape,
            *((1, ) * (x.dim() - log_snr.dim())
        ))

        # calculate the noised input
        alpha, sigma = log_snr.sigmoid().sqrt(), (-log_snr).sigmoid().sqrt()
        x_noised = x * alpha + epsilon * sigma

        # calculate the loss
        epsilon_hat = model(x_noised, t=t)
        loss = torch.nn.MSELoss()(epsilon_hat, epsilon)
        return loss
    
    def p_mean_variance(
            self,
            model: Callable,
            x: torch.Tensor,
            t: torch.Tensor,
            t_next: torch.Tensor,
    ):
        log_snr, log_snr_next = self.schedule(t), self.schedule(t_next)
        c = -torch.special.expm1(log_snr - log_snr_next)
        alpha, sigma = log_snr.sigmoid().sqrt(), (-log_snr).sigmoid().sqrt()
        alpha_next, sigma_next = log_snr_next.sigmoid().sqrt(), (-log_snr_next).sigmoid().sqrt() 
        t = t * torch.ones(x.size(0), device=x.device)
        epsilon_hat = model(x, t=t)
        x_start = (x - sigma * epsilon_hat) / alpha
        x_start.clamp_(min=-1, max=1)
        p_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        p_variance = sigma_next.pow(2) * c
        return p_mean, p_variance
    
    @torch.no_grad()
    def p_sample(
        self,
        model: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ):
        p_mean, p_variance = self.p_mean_variance(model, x, t, t_next)
        if t_next == 0:
            return p_mean
        epsilon = torch.randn_like(x)
        return p_mean + epsilon * p_variance.sqrt()
    
    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        shape: torch.Size,
    ):
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        steps = torch.linspace(
            1.0, 0.0, self.steps + 1, device=device
        )
        for i in range(self.steps):
            x = self.p_sample(model, x, steps[i], steps[i + 1])
        return x



