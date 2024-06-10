from typing import Optional
import torch
from .layers import RyeLayer
from .walk import generate_walk

class RyeModel(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_channels: int,
            length: int,
            repeat: int,
            layer: RyeLayer,
    ):
        super().__init__()
        self.layer = layer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_channels=num_channels,
        )
        self.length = length
        self.hidden_size = hidden_size
        self.repeat = repeat
        self.num_channels = num_channels
        self.input_size = input_size

    def forward(
            self,
            probability: torch.Tensor, # (N, N)
            invariant_input: torch.Tensor, # (N, input_size)
            equivariant_input: torch.Tensor, # (N, 3)
            invariant_hidden: Optional[torch.Tensor] = None, # (N, hidden_size)
            equivariant_hidden: Optional[torch.Tensor] = None, # (N, 3, num_channels)
    ):
        if invariant_hidden is None:
            invariant_hidden = torch.zeros(
                *invariant_input.shape[:-1], self.hidden_size, 
                device=invariant_input.device,
            )

        if equivariant_hidden is None:
            equivariant_hidden = torch.zeros(
                *equivariant_input.shape, self.num_channels,
                device=equivariant_input.device,
            )
        probability = probability.unsqueeze(-3).repeat_interleave(self.repeat, dim=-3)
        walks = generate_walk(probability, self.length).flip(-2)

        def indexing(_input, walks):
            return _input[walks]

        if invariant_input.dim() == 2:
            invariant_input = indexing(invariant_input, walks)
            equivariant_input = indexing(equivariant_input, walks)
        else:
            invariant_input = torch.vmap(indexing, in_dims=(0, 0))(invariant_input, walks)
            equivariant_input = torch.vmap(indexing, in_dims=(0, 0))(equivariant_input, walks)

        equivariant_hidden = equivariant_hidden.unsqueeze(-4).repeat_interleave(
            self.repeat, dim=-4
        )
        invariant_hidden = invariant_hidden.unsqueeze(-3).repeat_interleave(
            self.repeat, dim=-3,
        )

        invariant_hidden_traj, equivariant_hidden_traj = [], []
        for idx in range(self.length):
            if idx == 0:
                _equivariant_input = torch.zeros_like(equivariant_input[..., 0, :, :])
            else:
                _equivariant_input = equivariant_input[..., idx, :, :] \
                    - equivariant_input[..., idx-1, :, :]
                
            invariant_hidden, equivariant_hidden = self.layer(
                invariant_input[..., idx, :, :],
                _equivariant_input,
                invariant_hidden,
                equivariant_hidden,
            )
            invariant_hidden_traj.append(invariant_hidden)
            equivariant_hidden_traj.append(equivariant_hidden)
        invariant_hidden_traj = torch.stack(invariant_hidden_traj, dim=-3)
        equivariant_hidden_traj = torch.stack(equivariant_hidden_traj, dim=-4)

        return invariant_hidden_traj, equivariant_hidden_traj
        
        
        

    