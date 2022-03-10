import math
import torch
import torch.nn as nn

from termcolor import cprint
from collections import OrderedDict
import numpy as np

from model.nls.snl import SNLUnit
from model.nls.nl import NLUnit
from model.nls.ns import NSUnit
from model.nls.dnl import DNLUnit
from model.nls.arma import ARMAUnit
from model.nls.direct import DirectNLUnit
from model.nls.caylay import CayleySNLUnit

class Stage(nn.Module):
    def __init__(self, inplanes, planes, nl_type='snl', stage_num=5, aff_kernel='dot', is_sys=False, is_norm=False):
        super(Stage, self).__init__()
        self.down_channel = planes
        self.output_channel = inplanes
        self.num_block = stage_num
        self.input_channel = inplanes
        self.aff_kernel = aff_kernel
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.is_sys = is_sys
        self.is_norm = is_norm
        self.nl_type = nl_type

        if nl_type == 'snl':
            self.is_norm = True
            self.is_sys = True
        elif nl_type == 'arma':
            self.is_sys = True
            self.is_norm = True
        #    self.is_norm = False
        #    self.is_sys = True
        elif nl_type == 'dsnl':
            #self.is_norm = True
            self.is_norm = False
            self.is_sys = False
    

        layers = []
        for i in range(stage_num):
            if nl_type == 'snl':
                layers.append(SNLUnit(inplanes, planes))
            elif nl_type == 'dnl':
                layers.append(DNLUnit(inplanes, planes))
            elif nl_type == 'nl':
                layers.append(NLUnit(inplanes, planes))
            elif nl_type == 'a2':
                layers.append(NLUnit(inplanes, planes))
            elif nl_type == 'ns':
                layers.append(NSUnit(inplanes, planes))
            elif nl_type == 'arma':
                layers.append(ARMAUnit(inplanes, planes))
            elif nl_type == 'dsnl':
                layers.append(DirectNLUnit(inplanes, np.int64(planes / 2) ))
            elif nl_type == 'caylay':
                layers.append(CayleySNLUnit(inplanes, np.int64(planes / 2) ))

        self.stages = nn.Sequential(*layers)

    def DotKernel(self, x):

        t = self.t(x)
        p = self.p(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)

        if self.is_norm:
            att = torch.bmm(torch.relu(t), torch.relu(p))
            if self.is_sys:
                att = (att + att.permute(0, 2, 1)) / 2
            d = torch.sum(att, dim=2)
            d[d != 0] = torch.sqrt(1.0 / d[d != 0])
            #print(np.unique(np.array(d.cpu().data)))
            att = att * d.unsqueeze(1) * d.unsqueeze(2)
        else:
            if self.nl_type != 'a2' and self.nl_type != 'dsnl':
                att = torch.bmm(t, p)
            else:
                att = torch.bmm(torch.relu(t), torch.relu(p))

            if self.is_sys:
                att = (att + att.permute(0, 2, 1)) / 2

            if self.nl_type != 'a2' and self.nl_type != 'dsnl':
                att = torch.softmax(att, dim=2)
            
        return att


    def forward(self, x):

        att = self.DotKernel(x)

        if self.nl_type == 'dsnl':
            att_in = torch.bmm(att, att.permute(0, 2, 1)) 
            att_out = torch.bmm(att.permute(0, 2, 1), att) 
            att_sys = (att + att.permute(0, 2, 1)) / 2

            d_sys = torch.sum(att_sys, dim=2)
            d_sys[d_sys != 0] = torch.sqrt(1.0 / d_sys[d_sys != 0])
            att_sys = att_sys * d_sys.unsqueeze(1) * d_sys.unsqueeze(2)

            d_in = torch.sum(att_in, dim=2)
            d_in[d_in != 0] = torch.sqrt(1.0 / d_in[d_in != 0])
            att_in = att_in * d_in.unsqueeze(1) * d_in.unsqueeze(2)

            d_out = torch.sum(att_out, dim=2)
            d_out[d_out != 0] = torch.sqrt(1.0 / d_out[d_out != 0])
            att_out = att_out * d_out.unsqueeze(1) * d_out.unsqueeze(2)

        out = x

        for cur_stage in self.stages:
            if self.nl_type == 'dsnl':
                out = cur_stage(out, att_sys, att_in, att_out)
            else:
                out = cur_stage(out, att)

        return out








