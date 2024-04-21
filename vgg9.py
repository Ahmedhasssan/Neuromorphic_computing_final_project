"""
Spiking Neural Network with VGG architecture 
"""

import torch.nn as nn
from spikes import *
from layers.layers import SConv

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x

class VGGSNN_MINI(nn.Module):
    def __init__(self, num_classes):
        super(VGGSNN_MINI, self).__init__()
        self.num_classes = num_classes
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            SConv(1,8,3,1,1, wbit=2),
            pool,
            SConv(8,16,3,1,1, wbit=2),
            pool,
        )
        W = int(28/2/2)

        if num_classes ==10:
            self.classifier = SeqToANNContainer(nn.Linear(16*W*W,self.num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = add_dimention(input, self.T)
        #import pdb;pdb.set_trace()
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

def svgg9_mini(num_classes=10):
    model = VGGSNN_MINI(num_classes)
    return model