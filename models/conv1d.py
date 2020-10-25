import torch
from models import register_arch
import torch.nn as nn
from layers import base
from layers.classifiers import make_fc_layers
from layers import conv1d_encoder
from cpc.cpc import CPC
import logging
from cpc.metrics import CpcMetric


@register_arch
class Conv1dHcpc(nn.Module):
    def __init__(self,
                 feat,
                 n_mels,
                 sr,
                 in_bn,
                 d_model,
                 start_conv_groups,
                 start_conv_kernels,
                 start_conv_ds,
                 conv1d_blocks,
                 conv1d_block_type,
                 conv1d_block_ds,
                 conv1d_groups,
                 conv1d_dropout,
                 conv1d_bias,
                 conv1d_norm_before_res,
                 prediction_steps,
                 cpc_dropout,
                 cpc_hidden_units,
                 cpc_target_dim,
                 cpc_skip_context_num,
                 cpc_max_offset,
                 n_negatives,
                 softmax_tau,
                 is_random_crop,
                 cpc_list,
                 cpc_weight,
                 is_condition_cpc,
                 nb_classes,
                 fc_layers,
                 nonlinearity,
                 use_pre_ds_encoder,
                 **kwargs):
        super(Conv1dHcpc, self).__init__()
        feat_cls = getattr(base, feat)
        self.feat = feat_cls(n_mels=n_mels, sr=sr, bn=in_bn)
        if use_pre_ds_encoder:
            # max-pooling before conv in each block.
            encoder_cls = getattr(conv1d_encoder, 'PreDsConv1dEncoder')
        else:
            encoder_cls = getattr(conv1d_encoder, 'Conv1dEncoder')
        self.encoder = encoder_cls(d_in=n_mels,
                                   d_model=d_model,
                                   start_conv_kernels=start_conv_kernels,
                                   start_conv_ds=start_conv_ds,
                                   start_conv_groups=start_conv_groups,
                                   nonlinearity=nonlinearity,
                                   conv1d_groups=conv1d_groups,
                                   conv1d_blocks=conv1d_blocks,
                                   conv1d_block_type=conv1d_block_type,
                                   conv1d_block_ds=conv1d_block_ds,
                                   conv1d_dropout=conv1d_dropout,
                                   conv1d_bias=conv1d_bias,
                                   conv1d_norm_before_res=conv1d_norm_before_res)
        self.cpcs = nn.ModuleList()
        self.n_mels = n_mels
        self.is_random_crop = is_random_crop
        self.cpc_list = eval(cpc_list)
        self.d_model = d_model
        self.prediction_steps = prediction_steps
        self.n_negatives = n_negatives
        self.softmax_tau = softmax_tau
        self.cpc_weight = cpc_weight
        self._cpc_weight = cpc_weight
        self.is_condition_cpc = is_condition_cpc
        self.cpc_dropout = cpc_dropout
        self.cpc_hidden_units = cpc_hidden_units
        self.cpc_target_dim = cpc_target_dim
        self.cpc_skip_context_num = cpc_skip_context_num
        self.cpc_max_offset = cpc_max_offset
        self.n_classes = nb_classes
        self.fc_layers = make_fc_layers(d_in=self.d_model,
                                        fc_sizes=eval(fc_layers),
                                        nb_classes=nb_classes)

        if self.cpc_weight > 0:
            self.cpc_metrics = list()
            self.parse_cpc_list()
        self._log_num = 2

    def adjust_cpc_weight(self, current_epoch, warm_up):
        # train warm_up epochs using classification loss
        if current_epoch <= warm_up:
            self.encoder.cpc_detach = True
        else:
            self.encoder.cpc_detach = False

    def parse_cpc_list(self):
        for id, (i, j) in enumerate(self.cpc_list):

            # only apply conditional cpc on first cpc
            is_conditional_cpc = self.is_condition_cpc
            self.cpcs.append(CPC(d_z=self.n_mels if i == 0 else self.d_model,
                                 d_c=self.d_model,
                                 prediction_steps=self.prediction_steps,
                                 n_negatives=self.n_negatives,
                                 lrf=self.cpc_skip_context_num,
                                 causal_offset=min(self.encoder.offsets[i], self.cpc_max_offset),
                                 hop=int(self.encoder.hops[j] / self.encoder.hops[i]),
                                 softmax_tau=self.softmax_tau,
                                 is_conditional=is_conditional_cpc,
                                 cpc_dropout=self.cpc_dropout,
                                 cpc_hidden_units=self.cpc_hidden_units,
                                 cpc_target_dim=self.cpc_target_dim,
                                 cpc_id=id))
            self.cpc_metrics.append(CpcMetric(id=id, prediction_steps=self.prediction_steps))

    def _random_drop(self, x):
        """
        random drop leading frames up to 50 frames.
        """
        if self.training:
            shift = torch.randint(low=0, high=50, size=(1,), device='cuda')
            x = x[..., shift:]
        return x

    def forward(self, x):
        x = self.feat(x)
        if self.is_random_crop:
            x = self._random_drop(x)

        if self._log_num > 0:
            logging.info("x size after random_crop: {}".format(x.size()[1:]))
            self._log_num -= 1

        x = self.encoder(x)

        if self._log_num > 0:
            logging.info("x size after conv1d encoder: {}".format(x.size()[1:]))
            self._log_num -= 1

        if self.cpc_weight > 0:
            z = self.encoder.zlist

        for fc in self.fc_layers:
            x = fc(x)

        cpc_logits = list()
        cpc_labels = list()
        if self.cpc_weight > 0:
            for i, (start, end) in enumerate(self.cpc_list):
                logits, labels = self.cpcs[i](z[end], z[start])
                cpc_logits.append(logits)
                cpc_labels.append(labels)
                self.cpc_metrics[i].accumulate(logits, labels)

        return x, cpc_logits, cpc_labels



