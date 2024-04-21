"""
Customized quantization layers and modules

Example method:
SAWB-PACT: Accurate and Efficient 2-bit Quantized Neural Networks
RCF: Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
"""
import torch
import torch.nn.functional as F
from torch import Tensor        
import torch.nn as nn

class QBase(nn.Module):
    r"""Base quantization method for weight and activation.

    Args:
    nbit (int): Data precision.
    train_flag (bool): Training mode. 

    Attribute:
    dequantize (bool): Flag for dequantization (int -> descritized float).

    Methods:
    trainFunc (input:Tensor): Training function of quantization-aware training (QAT)
    evalFunc (input:Tensor): Forward pass function of inference. 
    inference(): Switch to inference mode. 
    """
    def __init__(self, nbit:int, train_flag:bool=True):
        super(QBase, self).__init__()
        self.nbit = nbit
        self.train_flag = train_flag
        self.dequantize = True
        self.qflag = True
    
    def q(self, input:Tensor):
        """
        Quantization operation
        """
        return input
    
    def trainFunc(self, input:Tensor):
        r"""Forward pass of quantization-aware training 
        """
        out = self.q(input)
        return out
    
    def evalFunc(self, input:Tensor):
        r"""Forward pass of inference
        """
        return self.trainFunc(input)
    
    def inference(self):
        r"""Inference mode
        """
        self.train_flag = False
        self.dequantize = False

    def forward(self, input:Tensor):
        if self.train_flag:
            y = self.trainFunc(input)
        else:
            y = self.evalFunc(input)
        return y
    
    def extra_repr(self) -> str:
        return super().extra_repr() + "nbit={}".format(self.nbit)

class QBaseConv2d(nn.Conv2d):
    r"""Basic low precision convolutional layer

    Inherited from the base nn.Conv2d layer.
    
    Args:
    wbit (int): Weight quantization precision. 
    abit (int): Input quantization precision.
    wq (QBase): Weight quantizer. 
    aq (QBase): Activation quantizer.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, wbit:int=32, abit:int=32, train_flag=True):
        super(QBaseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.train_flag = train_flag
        
        self.wbit = wbit
        self.abit = abit
        
        # quantizer
        self.wq = nn.Identity()
        self.aq = nn.Identity()
    
    def inference(self):
        """
        Inference mode.
        """
        self.train_flag = False
        self.register_buffer("qweight", torch.ones_like(self.weight))
        self.register_buffer("fm_max", torch.tensor(0.))

    def get_fm_info(self, y:Tensor):
        # maximum bit length
        mb = len(bin(int(y.abs().max().item()))) - 2
        fm = mb * y.size(2) * y.size(3)
        
        # maximum featuremap size
        if fm > self.fm_max:
            self.fm_max.data = torch.tensor(fm).float()

    def forward(self, input:Tensor):
        wq = self.wq(self.weight)

        xq = self.aq(input)
        y = F.conv2d(xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # save integer weights
        if not self.train_flag:
            self.qweight.data = wq
            self.get_fm_info(y)

        return y

class QBaseLinear(nn.Linear):
    r"""Basic low precision linear layer

    Inherited from the base nn.Linear layer.
    
    Args:
    wbit (int): Weight quantization precision. 
    abit (int): Input quantization precision.
    wq (QBase): Weight quantizer. 
    aq (QBase): Activation quantizer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, wbit:int=32, abit:int=32, train_flag=True):
        super(QBaseLinear, self).__init__(in_features, out_features, bias)
        self.train_flag = train_flag
        
        self.wbit = wbit
        self.abit = abit
        
        # quantizer
        self.wq = nn.Identity()
        self.aq = nn.Identity()
    
    def inference(self):
        """
        Inference mode
        """
        self.train_flag = False
    
    def forward(self, input:Tensor):
        wq = self.wq(self.weight)
        xq = self.aq(input)
        y = F.linear(xq, wq, self.bias)
        return y

class TernFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        tFactor = 0.05
        
        max_w = input.abs().max()
        th = tFactor*max_w #threshold
        output = input.clone().zero_()
        W = input[input.ge(th)+input.le(-th)].abs().mean()
        # output[input.ge(th)] = W
        # output[input.lt(-th)] = -W
        output[input.ge(th)] = 1
        output[input.lt(-th)] = -1

        return output
    @staticmethod
    def backward(ctx, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        return grad_input

class TernW(QBase):
    def __init__(self, nbit: int=2, train_flag: bool = True):
        super().__init__(nbit, train_flag)
        self.tFactor = 0.05
    
    def trainFunc(self, input: Tensor):
        #out = TernFunc.apply(input)
        levels = torch.tensor([-0.5,-0.25,0,0.25, 0.5,1])
        out = QMem.apply(input, levels)
        return out

class QMem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, levels):
        out = pqmem(inputs, levels, neg=levels.min().item())
        ctx.save_for_backward(inputs, levels)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, levels = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None

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

class QConv2d(QBaseConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, wbit: int = 32, abit: int = 32, train_flag=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        # layer index
        self.layer_idx = 0
        
        # quantizers
        if wbit < 32:
            self.wq = TernW(train_flag=True)
        

    def forward(self, input:Tensor):
        wq = self.wq(self.weight)
        y = F.conv2d(input, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

# class QLinear(QBaseLinear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, wbit: int = 32, abit: int = 32, train_flag=True):
#         super(QLinear, self).__init__(in_features, out_features, bias, wbit, abit, train_flag)

#         # quantizers
#         if wbit < 32:
#             if wbit in [4, 8]:
#                 self.wq = SAWB(self.wbit, train_flag=True, qmode="asymm")

#     def trainFunc(self, input):
#         return super().trainFunc(input)