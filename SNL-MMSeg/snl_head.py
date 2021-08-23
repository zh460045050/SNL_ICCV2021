import torch
from mmcv.cnn import NonLocal2d

from ..builder import HEADS
from .fcn_head import FCNHead
import numpy as np


@HEADS.register_module()
class SNLHead(FCNHead):
    """Non-local Neural Networks.

    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.

    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: True.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian.'.
    """
    

    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 **kwargs):
        super(SNLHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.SNL = SNLStage(
                    self.channels, np.int64(self.channels / self.reduction),
                    use_scale=False, stage_num=1,
                    relu=True, aff_kernel='dot')

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.SNL(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output

########
class SNLStage(nn.Module):
    def __init__(self, inplanes, planes, stage_num=5, use_scale=False, relu=False, aff_kernel='dot'):
        super(SNLStage, self).__init__()
        self.down_channel = planes
        self.output_channel = inplanes
        self.num_block = stage_num
        self.input_channel = inplanes
        #self.softmax = nn.Softmax(dim=2)
        self.aff_kernel = aff_kernel
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.use_scale = use_scale


        layers = []
        for i in range(stage_num):
            layers.append(SNLUnit(inplanes, planes, relu=relu))

        self.stages = nn.Sequential(*layers)
        self._init_params()


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def DotKernel(self, x):

        t = self.t(x)
        p = self.p(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)

        ####
        att = torch.bmm(torch.relu(t), torch.relu(p))
        att += att.permute(0, 2, 1)
        att = att / 2

        d = torch.sum(att, dim=2)
        d[d != 0] = torch.sqrt(1.0 / d[d != 0])
        att *= d.unsqueeze(1)
        att *= d.unsqueeze(2)
        ####
        
        return att


    def forward(self, x):

        if self.aff_kernel == 'dot':
            att = self.DotKernel(x)
        #elif self.aff_kernel == 'embedgassian':
        #    att = self.EbdedGassKernel(x)
        #elif self.aff_kernel == "gassian":
        #    att = self.GassKernel(x)
        #elif self.aff_kernel == 'rbf':
        #    att = self.RBFGassKeneral(x)
        else:
            raise KeyError("Unsupported nonlocal type: {}".format(nl_type))

        if self.use_scale:
            att = att.div(c**0.5)

        out = x

        for cur_stage in self.stages:
            out = cur_stage(out, att)

        return out



class SNLUnit(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, relu=False, aff_kernel='dot'):
        self.use_scale = use_scale

        super(SNLUnit, self).__init__()

        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.w_1 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.w_2 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)

        self.is_relu = relu
        if self.is_relu:
            self.relu = nn.ReLU(inplace=True)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, att):
        residual = x

        g = self.g(x)

        b, c, h, w = g.size()

        g = g.view(b, c, -1).permute(0, 2, 1)

        x_1 = g.permute(0, 2, 1).contiguous().view(b,c,h,w)
        x_1 = self.w_1(x_1)

        out = x_1

        x_2 = torch.bmm(att, g)
        x_2 = x_2.permute(0, 2, 1)
        x_2 = x_2.contiguous()
        x_2 = x_2.view(b, c, h, w)
        x_2 = self.w_2(x_2)
        out = out - x_2

        out = self.bn(out)

        if self.is_relu:
            out = self.relu(out)

        out = out + residual

        return out
#################################################################################################################
