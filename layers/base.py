import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def norm_layer1d(norm_type, num_channels, groups=1):
    assert norm_type in [None, 'layer_norm', 'group_norm', 'batch_norm']
    if norm_type == 'layer_norm':
        return nn.GroupNorm(num_groups=1, num_channels=num_channels)
    elif norm_type == 'batch_norm':
        return nn.BatchNorm1d(num_features=num_channels)
    elif norm_type == 'group_norm':
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)
    else:
        return None


def activation(nonlinearity='relu'):
    if nonlinearity == 'relu':
        return nn.ReLU()
    elif nonlinearity == 'gelu':
        return nn.GELU()
    elif nonlinearity == 'elu':
        return nn.ELU()
    else:
        return None


class MFCC(nn.Module):
    def __init__(self, sr, bn=False):
        super(MFCC, self).__init__()
        melspec_args = {
            'n_fft': 1024 if sr > 22050 else 512,
            'win_length': 1024 if sr > 22050 else 512,
            'hop_length': 512 if sr > 22050 else 256,
            'n_mels': 128}
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40, log_mels=True,
                                               melkwargs=melspec_args)

        self.bn = torchaudio.transforms.SlidingWindowCmn(cmn_window=300, min_cmn_window=100, center=True,
                                                         norm_vars=False) if bn else None

    def forward(self, x):
        # (B, C, T)
        x = self.mfcc(x)
        if self.bn:
            x = self.bn(x)
        return x


class LogMel(nn.Module):
    def __init__(self, sr=44100, n_mels=128, bn=False):
        super(LogMel, self).__init__()
        self.log_mel = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                 n_fft=1024 if sr > 22050 else 512,  # 23.2ms
                                                 win_length=1024 if sr > 22050 else 512,
                                                 hop_length=512 if sr > 22050 else 256,  # 11.6ms
                                                 n_mels=n_mels,
                                                 f_min=0.),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
        # TODO: check this works as wishes
        self.bn = nn.BatchNorm1d(num_features=n_mels) if bn else None

    def forward(self, x):
        # (B, C, T)
        x = self.log_mel(x)
        if self.bn:
            x = self.bn(x)
        return x


class LogMelNo(nn.Module):
    """
    No Overlapping, thus half the time dim.
    """
    def __init__(self, sr=44100, n_mels=128, bn=False):
        super(LogMelNo, self).__init__()
        self.log_mel = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                 n_fft=1024 if sr > 22050 else 512,  # 42.6ms
                                                 win_length=1024 if sr > 22050 else 512,
                                                 hop_length=1024 if sr > 22050 else 512,  # 10.6ms
                                                 n_mels=n_mels,
                                                 f_min=0.),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
        # TODO: check this works as wishes
        self.bn = nn.BatchNorm1d(num_features=n_mels) if bn else None

    def forward(self, x):
        # (B, C, T)
        x = self.log_mel(x)
        if self.bn:
            x = self.bn(x)
        return x


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class PermuteForTrans(nn.Module):
    def __init__(self):
        super(PermuteForTrans, self).__init__()

    def forward(self, x):
        # (B, D, T) -> (T, B, D)
        return x.permute([2, 0, 1])


class PermuteFromTrans(nn.Module):
    def __init__(self):
        super(PermuteFromTrans, self).__init__()

    def forward(self, x):
        # (T, B, D) -> (B, D, T)
        return x.permute([1, 2, 0])


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        """
        :param logits: (B, Class, [Optional] T, ...)
        :param target: (B, [Optional] T, ...)
        :return:
        """
        if self.smoothing > 0.:
            # one_hot = torch.zeros_like(logits).scatter_(dim=1, index=target.unsqueeze(1), src=torch.tensor(1.))
            one_hot = torch.zeros_like(logits).scatter_(dim=1, index=target.unsqueeze(1), src=torch.ones_like(logits))
            smoothed_target = one_hot * (1 - self.smoothing) + self.smoothing / logits.size(1)
            smoothed_target.requires_grad_(False)
            log_prob = F.log_softmax(logits, dim=1)
            loss = F.kl_div(input=log_prob, target=smoothed_target, reduction='sum')
        else:
            loss = F.cross_entropy(input=logits, target=target, reduction='sum')

        return loss


class CPCBCEWithLogitsLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, balanced=False):
        """
        :param smoothing:
        :param balanced: if balanced, negative samples loss divide by n_negatives.
        """
        super(CPCBCEWithLogitsLabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.balanced = balanced

    def forward(self, logits, target):
        """
        CPC predictions
        :param logits: (B, N+1)
        :param target: (B,), All zeros.
        :return:
        """
        n_neg = logits.size(-1) - 1
        # (B, 1)
        target_pos = torch.ones_like(target).unsqueeze(-1)
        # (B, N)
        target_neg = torch.zeros_like(target_pos).expand([-1, n_neg])
        # (B, N+1)
        target_pos_neg = torch.cat([target_pos, target_neg], dim=-1)

        # pretend to be a N+1 Multi-label binary classification.
        # positive samples have weight 1., negatives have weight 1. / N
        if self.balanced:
            pos_weight = torch.ones(target_pos_neg.size(-1))
            pos_weight[1:] = 1 / n_neg
        else:
            pos_weight = torch.ones(target_pos_neg.size(-1))

        if self.smoothing > 0:
            target_pos_neg = torch.abs(target_pos_neg - self.smoothing)

        target_pos_neg.requires_grad_(False)

        loss = F.binary_cross_entropy_with_logits(logits,
                                                  target_pos_neg,
                                                  reduction='sum',
                                                  pos_weight=pos_weight.to('cuda'))

        return loss


if __name__ == '__main__':
    m = CPCBCEWithLogitsLabelSmoothingLoss(smoothing=0.1, balanced=True)
    x = torch.randn(size=(64, 21)).to('cuda')
    label = torch.zeros(size=(64,)).to('cuda')
    l = m(x, label)
    print(l)
