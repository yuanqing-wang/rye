import torch
import dgl
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

    def model(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
    ):
        raise NotImplementedError

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

        # index into the walks
        def indexing(x, walks):
            return x[walks]

        batch_indexing = torch.vmap(indexing, in_dims = (0, 0))
        equivariant_input, invariant_input = batch_indexing(h, walks), batch_indexing(x, walks)

        model = lambda x, t: self.layer(x)
        loss = 0.0
        # run the model
        for idx in range(self.length):
            if idx == 0:
                _equivariant_input = torch.zeros_like(equivariant_input[..., 0, :, :])
            else:
                _equivariant_input = equivariant_input[..., idx, :, :] \
                    - equivariant_input[..., idx-1, :, :]
            
            invariant_input = invariant_input[..., idx, :, :]

            invariant_hidden, equivariant_hidden = model(
                invariant_input=invariant_input,
                equivariant_input=_equivariant_input,
                invariant_hidden=invariant_hidden,
                equivariant_hidden=equivariant_hidden,
            )


            _loss = self.diffusion.loss(
                # invariant_input[..., idx, :, :],
                model=model,
                x=_equivariant_input,
            )

            loss = loss + _loss
        return loss


            


