import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

# class ResNet34(nn.Module):
class SEResNet50(nn.Module):
    def __init__(self, pretrained):
        super(SEResNet50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["se_resnet50"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["se_resnet50"](pretrained=None)

        self.l0 = nn.Linear(2048, 168)
        self.l1 = nn.Linear(2048, 11)
        self.l2 = nn.Linear(2048, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0, l1, l2