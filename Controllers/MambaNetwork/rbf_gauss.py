import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter


class RBF_gaussian(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    centers: Tensor
    log_sigmas: Tensor

    def __init__(
        self, in_features: int, out_features, basis_func: int, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_func = basis_func
        self.centers = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.log_sigmas = Parameter(torch.empty(out_features, **factory_kwargs))
    
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        
        #init.kaiming_uniform_(self.centers, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)
        #init.kaiming_uniform_(self.centers, a=math.sqrt(5))
        init.xavier_uniform_(self.centers, gain=init.calculate_gain('leaky_relu', 0.2), generator=None)
        #init.uniform_(self.centers)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.centers)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.log_sigmas, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        size_exp = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size_exp)        
        c = self.centers.unsqueeze(0).expand(size_exp)       
        distances = (x - c).pow(2).sum(-1).pow(0.5) /torch.exp(self.log_sigmas).unsqueeze(0)
        #print(distances)
        return self.basis_func(distances)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


    def gaussian(alpha):
        
        phi = torch.exp(-1*alpha.pow(2))
        return phi
    
    def matern52(alpha):
        
        phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
        * alpha.pow(2))*torch.exp(-5**0.5*alpha)
        return phi

    def linear(alpha):
        phi = alpha
        return phi

    def quadratic(alpha):
        phi = alpha.pow(2)
        return phi

    def inverse_quadratic(alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
        return phi

    def multiquadratic(alpha):
        phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def inverse_multiquadratic(alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def spline(alpha):
        phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
        return phi

    def poisson_one(alpha):
        phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
        return phi

    def poisson_two(alpha):
        phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
        * alpha * torch.exp(-alpha)
        return phi

    def matern32(alpha):
        phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
        return phi

    

    def basis_func_dict():
        """
        A helper function that returns a dictionary containing each RBF
        """
        
        bases = {'gaussian': gaussian,
                'linear': linear,
                'quadratic': quadratic,
                'inverse quadratic': inverse_quadratic,
                'multiquadric': multiquadric,
                'inverse multiquadric': inverse_multiquadric,
                'spline': spline,
                'poisson one': poisson_one,
                'poisson two': poisson_two,
                'matern32': matern32,
                'matern52': matern52}
        return bases