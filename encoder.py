import torch.nn as nn
from torchvision.models import densenet161, resnet152, vgg19
import torch
import os
os.environ['TORCH_HOME']='/user_data/shaoanxi/models'
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, network='vgg19', config='Baseline'):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'densenet161':
            self.net = densenet161(pretrained=True)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        else:
            self.net = vgg19(pretrained=True)
            self.nets = list(self.net.features.children())
            if config == 'Focus':
                self.select_list = [4,36]
                self.dim = 512+64
            else:
                self.select_list = [36]
                self.dim = 512

    def forward(self, x):
        features = []
        for i in range(len(self.nets)):
            if i in self.select_list:
                feature = F.adaptive_avg_pool2d(x.clone(), 14)
                features.append(feature)
            x = self.nets[i](x)
        x = torch.cat(features, dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x
