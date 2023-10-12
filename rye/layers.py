import torch


class RyeCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_channels: int,
    ):
        super().__init__()
        self.ir = torch.nn.Linear(input_size, hidden_size)
        self.hr = torch.nn.Linear(hidden_size, hidden_size)
        self.iz = torch.nn.Linear(input_size, hidden_size)
        self.hz = torch.nn.Linear(hidden_size, hidden_size)
        self._in = torch.nn.Linear(input_size, hidden_size)
        self.hn = torch.nn.Linear(hidden_size, hidden_size)


    def forward(
            self,
            xi: torch.Tensor,
            yi: torch.Tensor,
            xh: torch.Tensor,
            yh: torch.Tensor,
    ):
        r = torch.sigmoid(self.ir(yi) + self.hr(yh))
        z = torch.sigmoid(self.iz(yi) + self.hz(yh))
        n = torch.tanh(self._in(yi) + r * self.hn(yh))
        y = (1 - z) * n + z * yh
        return y
        