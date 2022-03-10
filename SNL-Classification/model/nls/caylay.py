import math
import torch
import torch.nn as nn

from termcolor import cprint
from collections import OrderedDict
import numpy as np



class CayleySNLUnit(nn.Module):
    def __init__(self, inplanes, planes):

        super(CayleySNLUnit, self).__init__()

        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.w_1 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.w_2 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.w_h = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)

    def forward(self, x, att):

        residual = x

        g = self.g(x)

        b, c, h, w = g.size()

        g = g.view(b, c, -1).permute(0, 2, 1)

        x_1 = g.permute(0, 2, 1).contiguous().view(b,c,h,w)

        x_w = self.w_h(x_1)
        x_1 = self.w_1(x_1)
        I = torch.eye(att.size(1)).cuda().unsqueeze(0)
        
        L = I - att

        L_2 = torch.bmm(L, L)
        x_2 = torch.bmm(L_2, g)

        x_w = x_w.view(b, c, h*w).permute(0, 2, 1)
        x_2 = x_w + x_2

        x_2 = x_2.permute(0, 2, 1)
        x_2 = x_2.contiguous()
        x_2 = x_2.view(b, c, h, w)
        out = x_1 + self.w_2(x_2)

        out = self.bn(out)

        out = torch.relu(out)

        out = out + residual

        return out

        ####