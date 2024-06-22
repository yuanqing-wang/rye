import torch
import dgl
from functools import partial
from ..graph_walk import uniform_random_walk
from ..layers import RyeElman
from .diffusion import Diffusion

class DiffusionModel(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            num_samples: int = 1,
            length: int = 2,
            steps: int = 100,
    ):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features)
        self.layer = RyeElman(hidden_features, hidden_features, hidden_features)
        self.diffusion = Diffusion(steps=steps)
        self.num_samples = num_samples
        self.length = length

        # diffusion layers
        self.damping = torch.nn.Linear(hidden_features+2, hidden_features)
        self.project = torch.nn.Linear(hidden_features, hidden_features)
        self.hidden_features = hidden_features

    def model(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            invariant_hidden: torch.Tensor, # (N, hidden_features)
            equivariant_hidden: torch.Tensor, # (N, 3, num_channels)
    ):
        # project into hidden dimension
        t = torch.cat([t.cos(), t.sin()], dim=-1).unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], 2)
        invariant_hidden = torch.cat([invariant_hidden, t], dim=-1)
        damping = self.damping(invariant_hidden)
        project = self.project(equivariant_hidden)
        x = (x.unsqueeze(-1) + project) * damping.unsqueeze(-2)
        return x.mean(-1)
    
    def loss(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            x: torch.Tensor,
    ):
        # project into hidden dimension
        h = self.fc_in(h)

        # generate walks
        walks, _ = uniform_random_walk(
            g = g,
            length = self.length,
            num_samples = self.num_samples,
        )

        # initialize hidden features
        invariant_hidden = torch.zeros(
            *h.shape[:-1], self.hidden_features, 
            device=h.device,
        )

        equivariant_hidden = torch.zeros(
                *x.shape, self.hidden_features,
                device=x.device,
            )

        # index into the walks
        def indexing(x, walks):
            return x[walks]

        # batch the indexing
        batch_indexing = torch.vmap(indexing, in_dims = (0, 0))
        invariant_input, equivariant_input = batch_indexing(h, walks), batch_indexing(x, walks)

        # run the model
        for idx in range(self.length - 1):
            if idx == 0:
                _equivariant_input = torch.zeros_like(equivariant_input[..., 0, :])
            else:
                _equivariant_input = equivariant_input[..., idx, :] \
                    - equivariant_input[..., idx-1, :]
            
            invariant_input = invariant_input[..., idx, :]

            invariant_hidden, equivariant_hidden = self.layer(
                invariant_input, 
                _equivariant_input, 
                invariant_hidden, 
                equivariant_hidden,
            )

            model = partial(
                self.model,
                invariant_hidden=invariant_hidden,
                equivariant_hidden=equivariant_hidden,
            )


            loss = self.diffusion.loss( 
                model=model,
                x=equivariant_input[..., idx + 1, :],
            )



            


