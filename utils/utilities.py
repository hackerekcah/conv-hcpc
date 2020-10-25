import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import logging
import sys
import torch.nn.functional as F


def get_logger(log_file):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # logging to file
    fileh = logging.FileHandler(log_file, 'a')
    fileh.setFormatter(formatter)
    fileh.setLevel('INFO')

    # loggint to stdout
    streamh = logging.StreamHandler(sys.stdout)
    streamh.setLevel('INFO')
    streamh.setFormatter(formatter)

    logger = logging.getLogger()  # root logger
    logger.setLevel('INFO')

    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)

    logger.addHandler(fileh)
    logger.addHandler(streamh)
    return logger


def calculate_confusion_matrix(target, predict, classes_num):
    """Calculate confusion matrix.
    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes
    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num, classes_num))
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, title, labels, values):
    """Plot confusion matrix.
    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels
      values: list of values to be shown in diagonal
    Ouputs:
      None
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    if labels:
        ax.set_xticklabels([''] + labels, rotation=90, ha='left')
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for n in range(len(values)):
        plt.text(n - 0.4, n, '{:.2f}'.format(values[n]), color='yellow')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.tight_layout()
    plt.show()


def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.
    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
    Outputs:
      accuracy: float
    """

    samples_num = len(target)

    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')


def weighted_binary_cross_entropy(output, target, pos_weight=None, reduction='sum'):
    """
    :param output: prediction probabilities
    :param target:
    :param pos_weight: tensor with len same to number of class
    :param reduction:
    :return:
    """

    EPS = 1e-12

    if pos_weight is not None:
        assert len(pos_weight) == target.size(1)

        loss = pos_weight * (target * torch.log(output + EPS)) + ((1 - target) * torch.log(1 - output + EPS))
    else:
        loss = target * torch.log(output + EPS) + (1 - target) * torch.log(1 - output + EPS)

    if reduction == 'sum':
        return torch.neg(torch.sum(loss))
    elif reduction == 'mean':
        return torch.neg(torch.mean(loss))


class WeightedBCE:
    def __init__(self, pos_weight=None, reduction='sum'):
        self.pos_weight = pos_weight
        self.reduction = reduction

    def __call__(self, output, target):
        return weighted_binary_cross_entropy(output, target, pos_weight=self.pos_weight, reduction=self.reduction)


class CosineMarginProduct(torch.nn.Module):
    def __init__(self, in_feature=512, out_feature=10, s=30.0, m=0.35):
        super(CosineMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.Tensor(out_feature, in_feature))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, cosine, label):
        """
        :param input: (B, D)
        :param label: (B, 1) or (B,), torch.LongTensor?
        :return:
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        return output


class CosineMarginCrossEntropy(torch.nn.Module):
    def __init__(self, s=30.0, m=0.35):
        super(CosineMarginCrossEntropy, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        """
        :param input: (B, D)
        :param label: (B, 1) or (B,), torch.LongTensor?
        :return:
        """

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)

        loss = torch.nn.CrossEntropyLoss(reduction='sum')(output, label)

        return loss



