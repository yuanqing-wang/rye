from ast import Not
from typing import Optional
import torch
from typing import Tuple

INF = 1e6
EPSILON = 1e-6

class DotProductProjection(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc_left = torch.nn.Linear(input_size, output_size, bias=False)
        self.fc_right = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(
            self, 
            x0: torch.Tensor,
            x1: Optional[torch.Tensor] = None,
        ):
        if x1 is None:
            x1 = x0
        norm_left = x0.pow(2).sum(-2, keepdim=True) + EPSILON
        norm_right = x1.pow(2).sum(-2, keepdim=True) + EPSILON
        x0 = x0 / norm_left
        x1 = x1 / norm_right
        return (self.fc_left(x0) * self.fc_right(x1)).sum(-2)
    
class Damping(torch.nn.Module):
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
    

class RyeLayer(torch.nn.Module):
    def forward(
            self,
            invariant_input: torch.Tensor, # (N, input_size)
            equivariant_input: torch.Tensor, # (N, 3)
            invariant_hidden: torch.Tensor, # (N, hidden_size)
            equivariant_hidden: torch.Tensor, # (N, 3, num_channels)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class RyeElman(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_channels: int,
    ):
        super().__init__()
        self.equivariant_to_invariant = DotProductProjection(
            1 + num_channels, 
            hidden_size,
        )

        self.invariant_to_invariant = torch.nn.Linear(
            input_size + hidden_size, 
            hidden_size,
        )

        self.equivariant_to_equivariant = torch.nn.Linear(
            1 + num_channels, 
            num_channels,
            bias=False,
        )

        self.invariant_to_equivariant = Damping(
            hidden_size + input_size, 
            num_channels,
        )

        self.equivariant_output = torch.nn.Linear(
            num_channels,
            1,
        )

        self.invariant_output = torch.nn.Linear(
            hidden_size,
            hidden_size,
        )

    def forward(
            self,
            invariant_input: torch.Tensor, # (N, input_size)
            equivariant_input: torch.Tensor, # (N, 3)
            invariant_hidden: torch.Tensor, # (N, hidden_size)
            equivariant_hidden: torch.Tensor, # (N, 3, num_channels)
    ):

        # combine the input and the hidden equivariant
        # (N, 3, num_channels + 1)
        equivariant_combined = torch.cat(
            [
                equivariant_input.unsqueeze(-1), 
                equivariant_hidden
            ], 
            dim=-1,
        )

        # compute the input and the hidden invariant
        # (N, hidden_size + input_size)
        invariant_combined = torch.cat(
            [
                invariant_input, 
                invariant_hidden
            ], 
            dim=-1,
        )

        # compute invariant hidden
        # (N, hidden_size)
        invariant_hidden = torch.tanh(
            self.equivariant_to_invariant(equivariant_combined) \
            + self.invariant_to_invariant(invariant_combined)
        )

        # compute equivariant hidden
        equivariant_hidden = self.equivariant_to_equivariant(equivariant_combined) \
            + self.invariant_to_equivariant(equivariant_hidden, invariant_combined)
        
        return invariant_hidden, equivariant_hidden

        
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
        self.xn = Damping(hidden_size, 1)

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


def get_distance(x):
    delta_x = x.unsqueeze(-2) - x.unsqueeze(-3)
    distance = (delta_x ** 2).sum(-1)
    return distance

class RadialProbability(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x=None, distance=None):
        if distance is None:
            distance = get_distance(x)
        distance = distance / self.alpha
        distance.fill_diagonal_(INF)
        probability = (-distance).softmax(-1)
        return probability

class MeanReadout(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            activation,
            torch.nn.Linear(in_features, out_features),
        )

    def forward(self, x):
        return self.fc(x[..., -1, :, :].mean(-3))
