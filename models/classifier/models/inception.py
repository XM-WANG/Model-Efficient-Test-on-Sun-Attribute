from __future__ import absolute_import

import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable
import torch
from torchvision.models import inception_v3
from torch.nn import Parameter


class InceptionNet(nn.Module):
    def __init__(self,
                 pretrained=True,
                 num_class=102,
                 dropout=0,
                 input_size=(299, 299)):
        super(InceptionNet, self).__init__()
        self.pretrained = pretrained
        self.base = inception_v3(pretrained=pretrained)
        # Construct base (pretrained) InceptionNet
        self.num_class = num_class
        self.dropout = dropout

        out_planes = self.base.fc.in_features

        # Append new layers
        self.classifier = nn.Linear(out_planes, self.num_class)
        init.kaiming_normal(self.classifier.weight, mode='fan_out')
        init.constant(self.classifier.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            x = module(x)
        feature = x.view(x.size(0), -1)
        if self.dropout > 0:
            feature = self.drop(feature)
        output = self.classifier(feature)
        return output

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


if __name__ == '__main__':
    model = InceptionNet(num_class=102, input_size=(299, 299))
    x = Variable(torch.zeros(30, 3, 299, 299), requires_grad=False)
    output = model(x)
    print(output.data.size())
