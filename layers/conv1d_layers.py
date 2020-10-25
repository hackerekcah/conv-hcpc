import torch.nn as nn
import torch.nn.functional as F
import torch
from layers.base import activation
import logging
logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """
    residual before norm
    """
    def __init__(self, in_channels, out_channels, dropout, conv_bias, groups, block_ds, nonlinearity,
                 norm_before_res):
        super(BasicBlock, self).__init__()
        assert in_channels == out_channels

        self.conv1 = conv_norm_act(n_in=in_channels, n_out=out_channels, k=3, stride=1,
                                   conv_bias=conv_bias, groups=groups, nonlinearity=nonlinearity)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, stride=1, groups=groups, bias=conv_bias)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.block_ds = block_ds
        if self.block_ds:
            self.mp = CausalMaxPool1d(kernel_size=2, stride=2)
        self.activation = activation(nonlinearity=nonlinearity)
        self.norm_before_res = norm_before_res
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.norm_before_res:
            x = self.norm(x)
            x = self.drop(x)
            x += residual
        else:
            x = self.drop(x)
            x += residual
            x = self.norm(x)

        x = self.activation(x)

        if self.block_ds:
            x = self.mp(x)

        return x


class BottleneckBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, dropout, conv_bias, groups, nonlinearity,
                 block_ds,
                 norm_before_res):
        super(BottleneckBlock, self).__init__()
        width = int(in_channels / self.expansion)
        self.conv1 = CausalConv1d(in_channels=in_channels, out_channels=width, kernel_size=1, groups=groups,
                                  bias=conv_bias)
        self.norm1 = nn.GroupNorm(groups, width)
        self.act1 = activation(nonlinearity=nonlinearity)

        self.conv2 = CausalConv1d(in_channels=width, out_channels=width, kernel_size=3, groups=groups, bias=conv_bias)
        self.norm2 = nn.GroupNorm(groups, width)
        self.act2 = activation(nonlinearity=nonlinearity)

        self.conv3 = CausalConv1d(in_channels=width, out_channels=out_channels, kernel_size=1, groups=groups,
                                  bias=conv_bias)
        self.norm3 = nn.GroupNorm(groups, out_channels)
        self.act3 = activation(nonlinearity=nonlinearity)
        self.block_ds = block_ds
        if self.block_ds:
            self.mp = CausalMaxPool1d(kernel_size=2, stride=2)
        self.norm_before_res = norm_before_res
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)

        if self.norm_before_res:
            out = self.norm3(out)
            out = self.drop(out)
            out += identity
        else:
            out = self.drop(out)
            out += identity
            out = self.norm3(out)

        out = self.act3(out)

        if self.block_ds:
            out = self.mp(out)

        return out


class CoprimeBlock(nn.Module):
    """
    groups=3, combined with groups=2
    """
    def __init__(self, in_channels, out_channels, dropout, conv_bias, block_ds, nonlinearity, norm_before_res, groups):
        super(CoprimeBlock, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 6 == 0

        self.conv1 = conv_norm_act(n_in=in_channels, n_out=out_channels, k=3, stride=1,
                                   conv_bias=conv_bias, groups=3, nonlinearity=nonlinearity)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, stride=1, groups=2, bias=conv_bias)
        self.norm = nn.GroupNorm(1, out_channels)

        self.activation = activation(nonlinearity=nonlinearity)
        self.block_ds = block_ds
        if self.block_ds:
            self.mp = CausalMaxPool1d(kernel_size=2, stride=2)
        self.norm_before_res = norm_before_res
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)

        if self.norm_before_res:
            x = self.norm(x)
            x = self.drop(x)
            x += residual
        else:
            x = self.drop(x)
            x += residual
            x = self.norm(x)

        x = self.activation(x)

        if self.block_ds:
            x = self.mp(x)

        return x


def conv_norm_act(n_in, n_out, k, stride, conv_bias, groups, nonlinearity, dropout=0.):
    return nn.Sequential(
        CausalConv1d(n_in, n_out, k, stride=stride, bias=conv_bias, groups=groups),
        nn.Dropout(p=dropout),
        nn.GroupNorm(groups, n_out),
        activation(nonlinearity=nonlinearity),
    )


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation
        self.__stride = stride

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        # TODO: if stride != 1?
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :- int(self.__padding / self.__stride)]
        return result


class CausalMaxPool1d(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(CausalMaxPool1d, self).__init__()

        self.__padding = (kernel_size - 1)
        self.mp = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = F.pad(x, [self.__padding, 0])
        x = self.mp(x)
        return x