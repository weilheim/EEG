from __future__ import absolute_import

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from model import eegnet, lstm


def train(params):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', metavar='lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight-decay', metavar='wd', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--batch-size', metavar='bs', type=int, default=32,
                        help='batch size')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='whether shuffle EEG dataset or not')
    parser.add_argument('--seed', type=int, default=1)
    params = parser.parse_args()
    train(params)