import math
import torch
from torch import nn

def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s = 1 / weight.abs().mean().clamp(min=1e-5) ## clamp -> 최소값 보장
    result = (weight * s).round().clamp(-1, 1) / s 
    return result.type(dtype)

def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) / s
    return result.type(dtype)

class BitLinear(nn.Linear):
    ## layer = BitLinear(768, 768, bias=False) 768, 768 -> *kargs, bias=False -> **kwargs
    def __init__(self, *kargs, weight_bits=1, input_bits=8, **kwargs): 
        super(BitLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    '''
    def forward(self, input):
        quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
        quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()
        out = nn.functional.linear(quant_input, quant_weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out    
    '''

    def forward(self, input):
        quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()
        out = nn.functional.linear(input, quant_weight) ## activation quant deactivate
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out
