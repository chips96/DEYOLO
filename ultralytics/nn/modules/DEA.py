"""
DEA (DECA and DEPA) module
"""

import torch
import torch.nn as nn

from .conv import Conv


class DEA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        result_vi, result_ir = self.depa(self.deca(x))
        return self.act(result_vi + result_ir)


class DECA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        """convolution pyramid"""
        if p_kernel is None:
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel), nn.SiLU())
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, int(self.kernel_size/kernel1/kernel2), int(self.kernel_size/kernel1/kernel2), 0,
                      groups=channel),
            nn.SiLU()
        )

    def forward(self, x):
        b, c, h, w = x[0].size()
        w_vi = self.avg_pool(x[0]).view(b, c)
        w_ir = self.avg_pool(x[1]).view(b, c)
        w_vi = self.fc(w_vi).view(b, c, 1, 1)
        w_ir = self.fc(w_ir).view(b, c, 1, 1)

        glob_t = self.compress(torch.cat([x[0], x[1]], 1))
        glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t))) if min(h, w) >= self.kernel_size else torch.mean(
                                                                                    glob_t, dim=[2, 3], keepdim=True)
        result_vi = x[0] * (self.act(w_ir * glob)).expand_as(x[0])
        result_ir = x[1] * (self.act(w_vi * glob)).expand_as(x[1])

        return result_vi, result_ir


class DEPA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""
    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        self.conv1 = Conv(2, 1, 5)
        self.conv2 = Conv(2, 1, 5)
        self.compress1 = Conv(channel, 1, 3)
        self.compress2 = Conv(channel, 1, 3)
        self.act = nn.Sigmoid()

        """convolution merge"""
        if m_kernel is None:
            m_kernel = [3, 7]
        self.cv_v1 = Conv(channel, 1, m_kernel[0])
        self.cv_v2 = Conv(channel, 1, m_kernel[1])
        self.cv_i1 = Conv(channel, 1, m_kernel[0])
        self.cv_i2 = Conv(channel, 1, m_kernel[1])

    def forward(self, x):
        w_vi = self.conv1(torch.cat([self.cv_v1(x[0]), self.cv_v2(x[0])], 1))
        w_ir = self.conv2(torch.cat([self.cv_i1(x[1]), self.cv_i2(x[1])], 1))
        glob = self.act(self.compress1(x[0]) + self.compress2(x[1]))
        w_vi = self.act(glob + w_vi)
        w_ir = self.act(glob + w_ir)
        result_vi = x[0] * w_ir.expand_as(x[0])
        result_ir = x[1] * w_vi.expand_as(x[1])

        return result_vi, result_ir
