import torch.nn as nn
import torch
import torch.nn.functional as F
import logging


def make_fc_layers(d_in, fc_sizes, nb_classes):
    """
    :param d_in: encoder output size
    :param fc_sizes: list of int
    :param nb_classes:
    :return: ModuleList
    """
    classifier = nn.ModuleList()
    logging.info("fc layers {}".format(str(fc_sizes)))

    # insert d_in and nb_classes into fc_sizes
    fc_sizes.insert(0, d_in)
    fc_sizes.append(nb_classes)

    # list of tuples
    fc_tuples = [(i, j) for i, j in zip(fc_sizes[:-1], fc_sizes[1:])]

    for idx, (isz, osz) in enumerate(fc_tuples):
        # Last fc layer will not use bn and relu.
        if idx == len(fc_tuples) - 1:
            classifier.append(nn.Conv1d(in_channels=isz,
                                        out_channels=osz,
                                        kernel_size=1))
        else:
            classifier.append(nn.Sequential(
                nn.Conv1d(in_channels=isz,
                          out_channels=osz,
                          kernel_size=1),
                nn.BatchNorm1d(num_features=osz),
                nn.ReLU()
            ))
    return classifier


if __name__ == '__main__':
    m = make_fc_layers(d_in=256, fc_sizes=[], nb_classes=10)
    print(m)