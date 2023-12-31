from typing import Optional
import torch

class DotProductProjection(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc_left = torch.nn.Linear(input_size, output_size, bias=False)
        self.fc_right = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(
            self, 
            x0: torch.Tensor,
            x1: Optional[torch.Tensor],
        ):
        if x1 is None:
            x1 = x0
        return (self.fc_left(x0) * self.fc_right(x1)).sum(-2)
    
class Dampening(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(
            self, 
            x: torch.Tensor,
            y: torch.Tensor,
        ):
        y = self.fc(y).unsqueeze(-2)
        return x * y
    


class RyeRNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_channels: int,
    ):
        super().__init__()
        self.yi = torch.nn.Linear(input_size, hidden_size)
        self.yh = torch.nn.Linear(hidden_size, hidden_size)
        self.x = DotProductProjection(1+num_channels, hidden_size)

class RyeGRU(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_channels: int,
    ):
        super().__init__()
        self.yir = torch.nn.Linear(input_size, hidden_size)
        self.yhr = torch.nn.Linear(hidden_size, hidden_size)
        self.yiz = torch.nn.Linear(input_size, hidden_size)
        self.yhz = torch.nn.Linear(hidden_size, hidden_size)
        self.yin = torch.nn.Linear(input_size, hidden_size)
        self.yhn = torch.nn.Linear(hidden_size, hidden_size)

        self.xr = DotProductProjection(1+num_channels, hidden_size)
        self.xz = DotProductProjection(1+num_channels, hidden_size)
        self.xn = Dampening(hidden_size, 1)

    def forward(
            self,
            xi: torch.Tensor, # (N, 3)
            yi: torch.Tensor, # (N, input_size)
            xh: torch.Tensor, # (N, 3, num_channels)
            yh: torch.Tensor, # (N, hidden_size)
    ):
        # combine the input and the hidden vector
        # (N, 3, num_channels + 1)
        x_combined = torch.cat([xi.unsqueeze(-1), xh], dim=-1)

        # compute the reset and update gates
        # (N, hidden_size)
        r = torch.sigmoid(self.yir(yi) + self.yhr(yh) + self.xr(x_combined))
        z = torch.sigmoid(self.yiz(yi) + self.yhz(yh) + self.xz(x_combined))

        # compute the new scalar
        # (N, hidden_size)
        ny = torch.tanh(self.yin(yi) + r * self.yhn(yh))

        # compute the new vector
        nx = self.xn(x_combined, ny).squeeze(-1)
        y = (1 - z) * ny + z * yh
        return nx, y
        