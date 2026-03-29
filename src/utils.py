import math
import torch
from torch import nn

def weight_quant158(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s = 1 / weight.abs().mean().clamp(min=1e-5) ## clamp -> 최소값 보장
    result = (weight * s).round().clamp(-1, 1) / s 
    return result.type(dtype)

def weight_quant1(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s = 1 / weight.abs().mean().clamp(min=1e-5) ## clamp -> 최소값 보장
    sign_weight = torch.where(weight > 0, 1.0, -1.0)
    result = sign_weight / s
    return result.type(dtype)

class BitLinear158(nn.Linear):
    ## layer = BitLinear(768, 768, bias=False) 768, 768 -> *kargs, bias=False -> **kwargs
    def __init__(self, *kargs, weight_bits=1, input_bits=8, **kwargs): 
        super(BitLinear158, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, input):
        quant_weight = self.weight + (weight_quant158(self.weight, self.weight_bits) - self.weight).detach()
        out = nn.functional.linear(input, quant_weight) ## activation quant deactivate
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out


class BitLinear1(nn.Linear):
    ## layer = BitLinear(768, 768, bias=False) 768, 768 -> *kargs, bias=False -> **kwargs
    def __init__(self, *kargs, weight_bits=1, input_bits=8, **kwargs): 
        super(BitLinear1, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, input):
        quant_weight = self.weight + (weight_quant1(self.weight, self.weight_bits) - self.weight).detach()
        out = nn.functional.linear(input, quant_weight) ## activation quant deactivate
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out
