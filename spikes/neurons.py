import torch
import torch.nn as nn
from .qmem import power_quant, QMem, TernMem
from util import AverageMeter
import sys
from torch import Tensor

def heaviside(x:Tensor):
    return x.ge(0.).float()

############################################

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class QLIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0, lb=-30.0):
        super(QLIFSpike, self).__init__()
        self.act = ZIF.apply
        self.qfunc = QMem.apply

        self.thresh = thresh
        self.tau = tau
        self.gama = gama

        # membrane potential quantization
        ub = thresh + 1
        qrange = ub - lb
        self.interval = torch.tensor(1.0)
        self.levels = torch.tensor([lb + self.interval*i for i in range(int(qrange / self.interval))])

        # meters
        self.sr = AverageMeter()

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]

        for t in range(T):
            tmp = mem * self.tau
            mem = tmp + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)

            sr = spike.sum() / spike.numel()
            self.sr.update(sr)
    
            mem = (1 - spike) * mem
            
            # q mem
            mem = self.qfunc(mem, self.levels, self.interval)
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)