from __future__ import absolute_import
#from torchsummary import summary
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable
import torch
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torch.nn import Parameter
import warnings
warnings.filterwarnings('ignore')

class DenseNet(nn.Module):
    __factory = {
        '121': densenet121,
        '169': densenet169,
        '201': densenet201,
        '161': densenet161

    }

    def __init__(
            self, depth=121, pretrained=False, num_classes=102,
            dropout=0, bn_size=4, input_size=(256, 256)):
            super(DenseNet, self).__init__()
            self.num_class = num_classes
            self.bn_size = bn_size
            self.depth = depth
            self.pretrained = pretrained
            self.num_class = num_classes
            self.dropout = dropout
            self.base = DenseNet.__factory["{:d}".format(depth)](pretrained=pretrained,
                                                                 bn_size=4,
                                                                 drop_rate=dropout,
                                                                 num_classes=num_classes)
            n = {121: 1024, 169: 1664, 201: 1920, 161: 2208}
            out_planes = int(n[depth]*(input_size[0]/32)*(input_size[1]/32))
            #print(out_planes)
            
            # Add classifier
            self.classifier = nn.Linear(out_planes, self.num_class)
            init.kaiming_normal(self.classifier.weight, mode='fan_out')
            init.constant(self.classifier.bias, 0)
            
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            
            if not self.pretrained:
                self.reset_params()

    def forward(self, x):
        for name, module in self.base.features._modules.items():
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



if __name__ == "__main__":
    model = DenseNet(169, input_size=(128, 256))
    #x = Variable(torch.rand(30, 3, 256, 256), requires_grad=False)
    #output = model(x)
    #print(output.size)
    #summary(model, (3, 128, 256))




