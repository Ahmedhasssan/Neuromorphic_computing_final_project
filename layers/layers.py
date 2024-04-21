import torch.nn as nn
from spikes import SeqToANNContainer, QLIFSpike
from .qlayer import QConv2d

class SConv(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, pool=False, wbit=32, tau=0.5):
        super(SConv, self).__init__()
        if wbit < 32:
            self.fwd = SeqToANNContainer(
                QConv2d(in_plane, out_plane, kernel_size, stride, padding, wbit=wbit, abit=32),
                nn.BatchNorm2d(out_plane)
            )
        else:
            self.fwd = SeqToANNContainer(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
                nn.BatchNorm2d(out_plane)
            )
        self.act = QLIFSpike(tau=tau)

        if pool:
            self.pool = SeqToANNContainer(nn.AvgPool2d(2))
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.fwd(x)
        x = self.pool(x)
        x = self.act(x)
        return x