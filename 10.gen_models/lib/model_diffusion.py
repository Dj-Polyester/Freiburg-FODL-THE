import torch
import torch.nn as nn


class SimpleDiffusion(nn.Module):
    def __init__(self, nfeatures: int, nblocks: int = 2, nunits: int = 64):
        super(SimpleDiffusion, self).__init__()

        self.linear_projection = torch.nn.Linear(
            in_features=nfeatures+1, out_features=nunits)
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(nunits, nunits) for i in range(nblocks)])
        self.relu = torch.nn.ReLU()
        self.outblock = nn.Linear(nunits, nfeatures)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_cat = torch.hstack([x, t])
        input_projection = self.linear_projection(input_cat)
        for layer in self.linear_layers:
            input_projection = self.relu(layer(input_projection))
        output = self.outblock(input_projection)
        return output
