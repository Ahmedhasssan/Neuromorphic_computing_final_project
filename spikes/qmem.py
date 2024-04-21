"""
Membrane Potential Quantization
"""

import torch
from torch import Tensor
import torch.nn as nn

class SeqToANNContainer(nn.Module):
    """
    Adopted from SpikingJelly https://github.com/fangwei123456/spikingjelly
    """
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

def power_quant(x:Tensor, value_s):
    shape = x.shape
    xhard = x.view(-1)
    value_s = value_s.type_as(x)
    idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
    xhard = value_s[idxs].view(shape)
    return xhard

def pqmem(mem:Tensor, levels:Tensor, neg=-1.0, thresh=1.0):
    min_v = mem.min()
    mem = mem.clamp(min=min_v.round(), max=thresh)
    memq = power_quant(mem, levels)
    return memq

def sigmoid(x:Tensor, T:float=5.0, s=0.5):
    e = T * (x + s)
    return torch.sigmoid(e)

def dsigmoid(x:Tensor, T:float=5.0, s=0.5):
    sig = sigmoid(x, T, s)
    return T * (1-sig) * sig

class TernMem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, levels):
        out = pqmem(inputs, levels, neg=levels.min().item())
        ctx.save_for_backward(inputs)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        sg = dsigmoid(inputs, T=2.0, s=0.5) + dsigmoid(inputs, T=2.0, s=-0.5)
        grad_input = grad_input * sg
        return grad_input, None

class QMem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, levels, interval):
        out = pqmem(inputs, levels, neg=levels.min().item())
        ctx.save_for_backward(inputs, levels, interval)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, levels, interval = ctx.saved_tensors
        grad_input = grad_output.clone()

        sg = 0.0
        for i, l in enumerate(levels):
            shift = interval / 2
            
            if l != 0:
                s = l + shift if l < 0 else l - shift
                sg += dsigmoid(inputs, T=2.0, s=-s)
        
        grad_input = grad_input * sg
        return grad_input, None, None
