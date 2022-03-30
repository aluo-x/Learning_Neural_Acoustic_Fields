import torch
from torch import nn
import numpy as np

class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)

def distance(x1, x2):
    # by jacobrgardner
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res

# small improvement in performance
# @torch.jit.script
def fit_predict_torch(input_pos:torch.Tensor, input_target:torch.Tensor, predict_pos:torch.Tensor, bandwidth:torch.Tensor) -> torch.Tensor:
    dist_vector = -distance(predict_pos, input_pos)
    gauss_dist = torch.exp(dist_vector/(2.0 * torch.square(bandwidth.unsqueeze(0))))
    magnitude = torch.sum(gauss_dist, dim=1, keepdim=True)
    out = torch.mm(gauss_dist, input_target)/magnitude
    return out
