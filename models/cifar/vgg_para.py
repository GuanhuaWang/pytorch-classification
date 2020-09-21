'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

__all__ = ['vgg19_bn_para']


class VGG(nn.Module):

    def __init__(self, features, gpu_num = 2, num_classes=1000, split_size=64):
        super(VGG, self).__init__()
        self.split_size = split_size
        self.gpu_para = gpu_num
        self._initialize_weights()
        if gpu_num == 0:
            self.features = features.cuda("cuda:0")
            self.classifier = nn.Linear(512, num_classes).cuda("cuda:0")
        if gpu_num == 2:
            self.features_1 = features[0:27].cuda("cuda:0")
            self.features_2 = features[27:].cuda("cuda:1")
            self.classifier = nn.Linear(512, num_classes).cuda("cuda:1")
        elif gpu_num == 4:
            self.features_1 = features[0:14].cuda("cuda:0")
            self.features_2 = features[14:27].cuda("cuda:1")
            self.features_3 = features[27:40].cuda("cuda:2")
            self.features_4 = features[40:].cuda("cuda:3")
            self.classifier = nn.Linear(512, num_classes).cuda("cuda:3")
        elif gpu_num == 10:
            self.features_1 = features[0:3].cuda("cuda:0")
            self.features_2 = features[3:7].cuda("cuda:1")
            self.features_3 = features[7:10].cuda("cuda:2")
            self.features_4 = features[10:14].cuda("cuda:3")
            self.features_5 = features[14:20].cuda("cuda:4")
            self.features_6 = features[20:27].cuda("cuda:5")
            self.features_7 = features[27:33].cuda("cuda:6")
            self.features_8 = features[33:40].cuda("cuda:7")
            self.features_9 = features[40:49].cuda("cuda:8")
            self.features_10 = features[49:].cuda("cuda:9")
            self.classifier = nn.Linear(512, num_classes).cuda("cuda:9")
        elif gpu_num == 5:
            self.features_1 = features[0:10].cuda("cuda:0")
            self.features_2 = features[10:20].cuda("cuda:1")
            self.features_3 = features[20:30].cuda("cuda:2")
            self.features_4 = features[30:40].cuda("cuda:3")
            self.features_5 = features[40:].cuda("cuda:4")
            self.classifier = nn.Linear(512, num_classes).cuda("cuda:4")

    def forward(self, x):
        if self.gpu_para == 4:
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            s_prev = self.features_1(s_next).cuda("cuda:1")
            ret = []
            for s_next in splits:
                s_prev = self.features_2(s_prev).cuda("cuda:2")
                s_prev = self.features_3(s_prev).cuda("cuda:3")
                s_prev = self.features_4(s_prev)
                s_prev = s_prev.view(s_prev.size(0), -1)
                ret.append(self.classifier(s_prev))

                s_prev = self.features_1(s_next).cuda("cuda:1")
            s_prev = self.features_2(s_prev).cuda("cuda:2")
            s_prev = self.features_3(s_prev).cuda("cuda:3")
            s_prev = self.features_4(s_prev)
            s_prev = s_prev.view(s_prev.size(0), -1)
            ret.append(self.classifier(s_prev))
            x = torch.cat(ret)
        elif self.gpu_para == 0:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        elif self.gpu_para == 2:
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            s_prev = self.features_1(s_next).cuda("cuda:1")
            ret = []
            for s_next in splits:
                s_prev = self.features_2(s_prev)
                s_prev = s_prev.view(s_prev.size(0), -1)
                ret.append(self.classifier(s_prev))

                s_prev = self.features_1(s_next).cuda("cuda:1")
            s_prev = self.features_2(s_prev)
            s_prev = s_prev.view(s_prev.size(0), -1)
            ret.append(self.classifier(s_prev))
            x = torch.cat(ret)
        elif self.gpu_para == 10:
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            s_prev = self.features_1(s_next).cuda("cuda:1")
            ret = []
            for s_next in splits:
                s_prev = self.features_2(s_prev).cuda("cuda:2")
                s_prev = self.features_3(s_prev).cuda("cuda:3")
                s_prev = self.features_4(s_prev).cuda("cuda:4")
                s_prev = self.features_5(s_prev).cuda("cuda:5")
                s_prev = self.features_6(s_prev).cuda("cuda:6")
                s_prev = self.features_7(s_prev).cuda("cuda:7")
                s_prev = self.features_8(s_prev).cuda("cuda:8")
                s_prev = self.features_9(s_prev).cuda("cuda:9")
                s_prev = self.features_10(s_prev)
                s_prev = s_prev.view(s_prev.size(0), -1)
                ret.append(self.classifier(s_prev))

                s_prev = self.features_1(s_next).cuda("cuda:1")
            s_prev = self.features_2(s_prev).cuda("cuda:2")
            s_prev = self.features_3(s_prev).cuda("cuda:3")
            s_prev = self.features_4(s_prev).cuda("cuda:4")
            s_prev = self.features_5(s_prev).cuda("cuda:5")
            s_prev = self.features_6(s_prev).cuda("cuda:6")
            s_prev = self.features_7(s_prev).cuda("cuda:7")
            s_prev = self.features_8(s_prev).cuda("cuda:8")
            s_prev = self.features_9(s_prev).cuda("cuda:9")
            s_prev = self.features_10(s_prev)
            s_prev = s_prev.view(s_prev.size(0), -1)
            ret.append(self.classifier(s_prev))
            x = torch.cat(ret)
        elif self.gpu_para == 5:
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            s_prev = self.features_1(s_next).cuda("cuda:1")
            ret = []
            for s_next in splits:
                s_prev = self.features_2(s_prev).cuda("cuda:2")
                s_prev = self.features_3(s_prev).cuda("cuda:3")
                s_prev = self.features_4(s_prev).cuda("cuda:4")
                s_prev = self.features_5(s_prev)
                s_prev = s_prev.view(s_prev.size(0), -1)
                ret.append(self.classifier(s_prev))

                s_prev = self.features_1(s_next).cuda("cuda:1")
            s_prev = self.features_2(s_prev).cuda("cuda:2")
            s_prev = self.features_3(s_prev).cuda("cuda:3")
            s_prev = self.features_4(s_prev).cuda("cuda:4")
            s_prev = self.features_5(s_prev)
            s_prev = s_prev.view(s_prev.size(0), -1)
            ret.append(self.classifier(s_prev))
            x = torch.cat(ret)
        else:
            x = self.features_1(x)
            x = self.features_2(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, gpu_para, batch_norm=False):
    layers = []
    in_channels = 3
    for index, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg19_bn_para(num_classes=10, gpu_num = 2, split_size=64):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], 2, batch_norm=True), gpu_num = gpu_num, num_classes=num_classes, split_size=split_size)
    return model