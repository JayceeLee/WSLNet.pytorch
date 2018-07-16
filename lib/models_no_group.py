# feature add before class-wise pooling
# not no group
# the new version is with feat attention
import torch 
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from .pooling import WildcatPool2d, ClassWisePool, ClassWisePool_avg
from .layers import AttentionLayer

class ResNetWSL(nn.Module):

    def __init__(self, model, num_classes, num_maps, pooling=None):
        super(ResNetWSL, self).__init__()

        self.num_maps = num_maps
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels

        # self.de_conv = nn.ConvTranspose2d(num_features, num_classes, kernel_size=3, stride=1, padding=0, bias=True)
        # self.de_conv = nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.group_conv = nn.Conv2d(num_features, num_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.attnlayer = AttentionLayer(num_features, num_classes)

        self.class_pooling = pooling.class_wise
        self.spatial_pooling = pooling.spatial

        self.class_pooling_avg = pooling.class_wise_avg

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
    
    def forward(self, x):
        x = self.features(x)
        y = self.attnlayer(x)
        # y = self.splayer(x)
        # b, c, _, _ = y.size()
        # b, c, _, _ = x.size()
        # y = y.repeat(1, 1, 1, self.num_maps).view(b, c*self.num_maps, 1, 1)
        # x = self.classifier(x)
        # x = self.de_conv(x)
        x = self.group_conv(x)
        # x = x + y*x

        # x = self.group_conv(x)
        # y = self.class_pooling_avg(x)
        x = self.class_pooling(x)
        b, c, _, _ = x.size() 

        # x = self.non_local_layer(x)

        # y = self.attnlayer(y)
        x = x + y*x
        x = self.spatial_pooling(x)
        # x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(b, c)
        return F.sigmoid(x)
    

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                # {'params': self.de_conv.parameters()},
                {'params': self.group_conv.parameters()},
                {'params': self.attnlayer.parameters()},
                {'params': self.class_pooling.parameters()},
                {'params': self.class_pooling_avg.parameters()},
                {'params': self.spatial_pooling.parameters()}]

def resnet50_no_group(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.resnet50(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('class_wise_avg', ClassWisePool_avg(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return ResNetWSL(model, num_classes, num_maps, pooling=pooling)

def resnet101_no_group(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.resnet101(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('class_wise_avg', ClassWisePool_avg(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return ResNetWSL(model, num_classes, num_maps, pooling=pooling)
