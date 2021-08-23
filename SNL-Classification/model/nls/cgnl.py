import math
import torch
import torch.nn as nn

from termcolor import cprint
from collections import OrderedDict
import numpy as np

class SpatialCGNLx(nn.Module):
    """Spatial CGNL block with Gaussian RBF kernel for image classification.
    """
    def __init__(self, inplanes, planes, groups=1, order=2, is_sys=True, is_norm=True):
        self.groups = groups
        self.order = order
        self.is_sys = is_sys
        self.is_norm = is_norm

        super(SpatialCGNLx, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                                                  groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.groups > 1:
            cprint("=> WARN: SpatialCGNLx block uses '{}' groups".format(self.groups), \
                   'yellow')

        cprint('=> WARN: The Taylor expansion order in SpatialCGNLx block is {}'.format(self.order), \
               'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The non-linear kernel (Gaussian RBF).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """

        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        # gamma
        gamma = torch.Tensor(1).fill_(1e-4)

        # NOTE:
        # We want to keep the high-order feature spaces in Taylor expansion to
        # rich the feature representation, so the l2 norm is not used here.
        #
        # Under the above precondition, the β should be calculated
        # by β = exp(−γ(∥θ∥^2 +∥φ∥^2)).
        # But in the experiments, we found training becomes very difficult.
        # So we simplify the implementation to
        # ease the gradient computation through calculating the β = exp(−2γ).

        # beta
        beta = torch.exp(-2 * gamma)

        t_taylor = []
        p_taylor = []
        for order in range(self.order+1):
            # alpha
            alpha = torch.mul(
                    torch.div(
                    torch.pow(
                        (2 * gamma),
                        order),
                        math.factorial(order)),
                        beta)

            alpha = torch.sqrt(
                        alpha.cuda())

            _t = t.pow(order).mul(alpha)
            _p = p.pow(order).mul(alpha)

            t_taylor.append(_t)
            p_taylor.append(_p)

        t_taylor = torch.cat(t_taylor, dim=1)
        p_taylor = torch.cat(p_taylor, dim=1)

        att = torch.bmm(p_taylor, g)

        att = att.view(b, 1, int(self.order+1))
        x = torch.bmm(att, t_taylor)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x