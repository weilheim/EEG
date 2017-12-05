from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """2D convolutional neural network for single EEG frame."""
    VALIDENDPOINT = ('logit', 'predict')

    def __init__(self, num_class,
                 input_channel,
                 hidden_size,
                 kernel_size,
                 stride,
                 avgpool_size=4,
                 dropout=0.1):
        super(EEGNet, self).__init__()

        assert len(kernel_size) == len(hidden_size)
        assert len(kernel_size) == len(stride)
        self.num_layer = len(kernel_size)
        self.num_class = num_class
        self.input_channel = input_channel
        self.dropout = dropout

        in_channel = self.input_channel
        layer = 1
        self.projections = nn.ModuleList()
        self.residualnorms = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        for (out_channel, kernel_width, s) in zip(hidden_size, kernel_size, stride):
            pad = (kernel_size - 1) // 2
            self.projections.append(nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=s, bias=False)
                                    if in_channel != out_channel or s != 1 else None)
            self.residualnorms.append(nn.BatchNorm2d(out_channel)
                                      if in_channel != out_channel or s != 1 else None)
            self.convolutions.append(nn.Conv2d(in_channel, out_channel,
                                               kernel_size=kernel_width, stride=s, padding=pad, bias=False))
            self.batchnorms.append(nn.BatchNorm2d(out_channel, eps=1e-5, affine=True))
            in_channel = out_channel
        # avgpool_size should equal to size of the feature map,
        # otherwise self.predict will break.
        self.avgpool = nn.AvgPool2d(kernel_size=avgpool_size)
        self.predict = nn.Linear(hidden_size[-1], self.num_class, bias=True)

    def forward(self, x, endpoint='predict'):
        if endpoint not in self.VALIDENDPOINT:
            raise ValueError('Unknown endpoint {:s}'.format(endpoint))

        for proj, rbn, conv, bn in zip(self.projections, self.residualnorms,
                                  self.convolutions, self.batchnorms):
            if proj is not None:
                residual = proj(x)
                residual = rbn(residual)
            else:
                residual = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = bn(x)
            x = (x + residual)
            x = F.relu(x)
        x = self.avgpool(x)
        if endpoint == 'logit':
            return x
        x = self.predict(x)
        return x


