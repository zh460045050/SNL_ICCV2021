import math
import torch
import torch.nn as nn

from termcolor import cprint
from collections import OrderedDict
import numpy as np



class NLUnit(nn.Module):
    def __init__(self, inplanes, planes):

        super(NLUnit, self).__init__()

        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.w_2 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)

    def forward(self, x, att):
        residual = x

        g = self.g(x)

        b, c, h, w = g.size()

        g = g.view(b, c, -1).permute(0, 2, 1)

        x_2 = torch.bmm(att, g)
        x_2 = x_2.permute(0, 2, 1)
        x_2 = x_2.contiguous()
        x_2 = x_2.view(b, c, h, w)
        x_2 = self.w_2(x_2)

        out = self.bn(x_2)
        out = torch.relu(out)
        out = out + residual

        return out


####