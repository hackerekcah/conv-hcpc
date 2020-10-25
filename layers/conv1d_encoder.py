import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from layers.conv1d_layers import conv_norm_act, CausalMaxPool1d
import logging
from layers import conv1d_layers
logger = logging.getLogger(__name__)


class Conv1dEncoder(nn.Module):
    def __init__(self, d_in,
                 d_model,
                 start_conv_kernels,
                 start_conv_ds,
                 start_conv_groups,
                 nonlinearity,
                 conv1d_groups,
                 conv1d_blocks,
                 conv1d_block_type,
                 conv1d_block_ds,
                 conv1d_dropout,
                 conv1d_bias,
                 conv1d_norm_before_res
                 ):
        super(Conv1dEncoder, self).__init__()
        self.cpc_detach = False
        self.zsizes = []
        self.hops = []
        self.offsets = []
        self.zsizes.append(d_in)
        _lrf = 1
        _hop = 1
        self.offsets.append(math.ceil(_lrf / _hop))
        self.hops.append(_hop)

        self.start_conv = []
        self.start_conv.append(conv_norm_act(n_in=d_in,
                                             n_out=d_model,
                                             k=start_conv_kernels,
                                             stride=1,
                                             conv_bias=conv1d_bias,
                                             groups=start_conv_groups,
                                             nonlinearity=nonlinearity))

        _lrf = (start_conv_kernels - 1) * _hop + _lrf
        _hop = _hop

        if start_conv_ds:
            self.start_conv.append(CausalMaxPool1d(kernel_size=2, stride=2))
            _lrf = _lrf + _hop * 1
            _hop = _hop * 2

        self.start_conv = nn.Sequential(*self.start_conv)

        self.zsizes.append(d_model)
        self.offsets.append(math.ceil(_lrf / _hop))
        self.hops.append(_hop)

        self.layers = nn.ModuleList()
        block_cls = getattr(conv1d_layers, conv1d_block_type)
        for i in range(conv1d_blocks):
            self.layers.append(block_cls(
                in_channels=d_model, out_channels=d_model, dropout=conv1d_dropout,
                conv_bias=conv1d_bias, groups=conv1d_groups, block_ds=conv1d_block_ds, nonlinearity=nonlinearity,
                norm_before_res=conv1d_norm_before_res
            ))
            # conv
            _lrf = _lrf + (3 - 1) * _hop
            _hop = _hop
            # conv
            _lrf = _lrf + (3 - 1) * _hop
            _hop = _hop
            # pool
            _lrf = _lrf + 1 * _hop
            _hop = _hop * 2

            self.zsizes.append(d_model)
            self.offsets.append(math.ceil(_lrf / _hop))
            self.hops.append(_hop)

        logger.info("offsets: {}".format(self.offsets))
        logger.info("hops: {}".format(self.hops))
        logger.info("zsizes: {}".format(self.zsizes))

    def forward(self, x):
        self.zlist = list()
        self.zlist.append(x) if not self.cpc_detach else self.zlist.append(x.detach())

        x = self.start_conv(x)
        self.zlist.append(x) if not self.cpc_detach else self.zlist.append(x.detach())

        for layer in self.layers:
            x = layer(x)
            self.zlist.append(x) if not self.cpc_detach else self.zlist.append(x.detach())

        return x


class PreDsConv1dEncoder(nn.Module):
    """
    get z as features before downsample.
    """
    def __init__(self, d_in,
                 d_model,
                 start_conv_kernels,
                 start_conv_ds,
                 start_conv_groups,
                 nonlinearity,
                 conv1d_groups,
                 conv1d_blocks,
                 conv1d_block_type,
                 conv1d_block_ds,
                 conv1d_dropout,
                 conv1d_bias,
                 conv1d_norm_before_res
                 ):
        super(PreDsConv1dEncoder, self).__init__()
        self.cpc_detach = False
        self.zsizes = []
        self.hops = []
        self.offsets = []
        self.zsizes.append(d_in)
        _lrf = 1
        _hop = 1
        self.offsets.append(math.ceil(_lrf / _hop))
        self.hops.append(_hop)

        self.start_conv = conv_norm_act(n_in=d_in,
                                        n_out=d_model,
                                        k=start_conv_kernels,
                                        stride=1,
                                        conv_bias=conv1d_bias,
                                        groups=start_conv_groups,
                                        nonlinearity=nonlinearity)

        _lrf = (start_conv_kernels - 1) * _hop + _lrf
        _hop = _hop

        self.offsets.append(math.ceil(_lrf / _hop))
        self.hops.append(_hop)
        self.zsizes.append(d_model)

        if start_conv_ds:
            self.start_conv_ds_layer = CausalMaxPool1d(kernel_size=2, stride=2)
            _lrf = _lrf + _hop * 1
            _hop = _hop * 2

        self.layers = nn.ModuleList()
        self.ds_layers = nn.ModuleList()
        block_cls = getattr(conv1d_layers, conv1d_block_type)
        for i in range(conv1d_blocks):
            self.layers.append(block_cls(
                in_channels=d_model, out_channels=d_model, dropout=conv1d_dropout,
                conv_bias=conv1d_bias, groups=conv1d_groups, block_ds=False, nonlinearity=nonlinearity,
                norm_before_res=conv1d_norm_before_res
            ))
            # conv
            _lrf = _lrf + (3 - 1) * _hop
            _hop = _hop
            # conv
            _lrf = _lrf + (3 - 1) * _hop
            _hop = _hop

            self.zsizes.append(d_model)
            self.offsets.append(math.ceil(_lrf / _hop))
            self.hops.append(_hop)

            if conv1d_block_ds:
                self.ds_layers.append(CausalMaxPool1d(kernel_size=2, stride=2))
                # pool
                _lrf = _lrf + 1 * _hop
                _hop = _hop * 2

        self.conv1d_block_ds = conv1d_block_ds
        self.start_conv_ds = start_conv_ds

        logger.info("offsets: {}".format(self.offsets))
        logger.info("hops: {}".format(self.hops))
        logger.info("zsizes: {}".format(self.zsizes))

    def forward(self, x):
        self.zlist = list()
        self.zlist.append(x) if not self.cpc_detach else self.zlist.append(x.detach())

        x = self.start_conv(x)
        self.zlist.append(x) if not self.cpc_detach else self.zlist.append(x.detach())
        if self.start_conv_ds:
            x = self.start_conv_ds_layer(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            self.zlist.append(x) if not self.cpc_detach else self.zlist.append(x.detach())
            if self.conv1d_block_ds:
                # downsample
                x = self.ds_layers[i](x)
        return x