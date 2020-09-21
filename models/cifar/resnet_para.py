from __future__ import absolute_import

import torch.nn as nn
import math
import torch


__all__ = ['resnet_para', 'resnet164_para']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, output_gpu=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.output_gpu = output_gpu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.output_gpu is not None:
            out = self.relu(out).cuda("cuda:"+str(self.output_gpu))
        else:
            out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock', split_size=0, gpu_num=1):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')
        self.gpu_num = gpu_num
        self.split_size = split_size
        self.inplanes = 16
        ### Split model into specific gpu. Each layer has 18 Bottleneck
        if self.gpu_num == 2:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                bias=False).cuda("cuda:0")
            self.bn1 = nn.BatchNorm2d(16).cuda("cuda:0")
            self.relu = nn.ReLU(inplace=True).cuda("cuda:0")
            self.layer1 = self._make_layer(block, 16, n, layer_index=1)
            self.layer2 = self._make_layer(block, 32, n, stride=2, layer_index=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2, layer_index=3)
            self.avgpool = nn.AvgPool2d(8).cuda("cuda:1")
            self.fc = nn.Linear(64 * block.expansion, num_classes).cuda("cuda:1")
        elif self.gpu_num == 4:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                bias=False).cuda("cuda:0")
            self.bn1 = nn.BatchNorm2d(16).cuda("cuda:0")
            self.relu = nn.ReLU(inplace=True).cuda("cuda:0")
            self.layer1 = self._make_layer(block, 16, n, layer_index=1).cuda("cuda:1")
            self.layer2 = self._make_layer(block, 32, n, stride=2, layer_index=2).cuda("cuda:2")
            self.layer3 = self._make_layer(block, 64, n, stride=2, layer_index=3).cuda("cuda:3")
            self.avgpool = nn.AvgPool2d(8).cuda("cuda:3")
            self.fc = nn.Linear(64 * block.expansion, num_classes).cuda("cuda:3")    
        elif self.gpu_num == 5:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                bias=False).cuda("cuda:0")
            self.bn1 = nn.BatchNorm2d(16).cuda("cuda:0")
            self.relu = nn.ReLU(inplace=True).cuda("cuda:0")
            self.layer1 = self._make_layer(block, 16, n, layer_index=1)
            self.layer2 = self._make_layer(block, 32, n, stride=2, layer_index=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2, layer_index=3)
            self.avgpool = nn.AvgPool2d(8).cuda("cuda:4")
            self.fc = nn.Linear(64 * block.expansion, num_classes).cuda("cuda:4") 
        elif self.gpu_num == 10:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                bias=False).cuda("cuda:0")
            self.bn1 = nn.BatchNorm2d(16).cuda("cuda:0")
            self.relu = nn.ReLU(inplace=True).cuda("cuda:0")
            self.layer1 = self._make_layer(block, 16, n, layer_index=1)
            self.layer2 = self._make_layer(block, 32, n, stride=2, layer_index=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2, layer_index=3)
            self.avgpool = nn.AvgPool2d(8).cuda("cuda:9")
            self.fc = nn.Linear(64 * block.expansion, num_classes).cuda("cuda:9")           
        else:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n, layer_index=1)
            self.layer2 = self._make_layer(block, 32, n, stride=2, layer_index=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2, layer_index=3)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, layer_index=None):  ### layer_index can only be layer1, layer2, layer3, 
        downsample = None                                                      ###each layer has 18 blocks
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        #layers = []
        if layer_index == 1:
            layers = self.make_layer_1(block, planes, blocks, stride, downsample)
        elif layer_index == 2:
            layers = self.make_layer_2(block, planes, blocks, stride, downsample)
        elif layer_index == 3:
            layers = self.make_layer_3(block, planes, blocks, stride, downsample)
        else:
            raise ValueError('You are not making resnet164 !!!')
        return nn.Sequential(*layers)


    def make_layer_1(self, block, planes, blocks, stride, downsample):
        layers = []
        if self.gpu_num == 2:
            device = "cuda:0"
            layers.append(block(self.inplanes, planes, stride, downsample, output_gpu=0).cuda(device))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, output_gpu=0).cuda(device))
        elif self.gpu_num == 5:
            device = "cuda:1"
            layers.append(block(self.inplanes, planes, stride, downsample, output_gpu=1).cuda(device))
            self.inplanes = planes * block.expansion
            output_loc = int(device[-1])
            for i in range(1, blocks):
                if i == 13:
                    output_loc = int(device[-1])+1
                if i == 14:
                    device = "cuda:2"
                layers.append(block(self.inplanes, planes, output_gpu=output_loc).cuda(device))
        elif self.gpu_num == 10:
            device = "cuda:1"
            layers.append(block(self.inplanes, planes, stride, downsample, output_gpu=1).cuda(device))
            self.inplanes = planes * block.expansion
            output_loc = int(device[-1])
            for i in range(1, blocks):
                if i == 5:
                    output_loc = int(device[-1])+1
                if i == 6:
                    device = "cuda:2"
                    output_loc = int(device[-1])
                if i == 11:
                    output_loc = int(device[-1])+1
                if i == 12:
                    device = "cuda:3"
                    output_loc = int(device[-1])
                if i == 17:
                    output_loc = int(device[-1])+1
                layers.append(block(self.inplanes, planes, output_gpu=output_loc).cuda(device))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        return layers
    
    def make_layer_2(self, block, planes, blocks, stride, downsample):
        layers = []
        if self.gpu_num == 10:
            device = "cuda:4"
            layers.append(block(self.inplanes, planes, stride, downsample, output_gpu=4).cuda(device))
            self.inplanes = planes * block.expansion
            output_loc = int(device[-1])
            for i in range(1, blocks):
                if i == 5:
                    output_loc = int(device[-1])+1
                if i == 6:
                    device = "cuda:5"
                    output_loc = int(device[-1])
                if i == 11:
                    output_loc = int(device[-1])+1
                if i == 12:
                    device = "cuda:6"
                    output_loc = int(device[-1])
                if i == 17:
                    output_loc = int(device[-1])+1
                layers.append(block(self.inplanes, planes, output_gpu=output_loc).cuda(device))
        elif self.gpu_num == 5:
            device = "cuda:2"
            layers.append(block(self.inplanes, planes, stride, downsample, output_gpu=2).cuda(device))
            self.inplanes = planes * block.expansion
            output_loc = int(device[-1])
            for i in range(1, blocks):
                if i == 9:
                    output_loc = int(device[-1])+1
                if i == 10:
                    device = "cuda:3"
                layers.append(block(self.inplanes, planes, output_gpu=output_loc).cuda(device))
        elif self.gpu_num == 2:
            device = "cuda:1"
            layers.append(block(self.inplanes, planes, stride, downsample, output_gpu=1).cuda(device))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, output_gpu=1).cuda(device))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        return layers
    
    def make_layer_3(self, block, planes, blocks, stride, downsample):
        layers = []
        device = "cuda:7"
        if self.gpu_num == 10:
            layers.append(block(self.inplanes, planes, stride, downsample, output_gpu=7).cuda(device))
            self.inplanes = planes * block.expansion
            output_loc = int(device[-1])
            for i in range(1, blocks):
                if i == 5:
                    output_loc = int(device[-1])+1
                if i == 6:
                    device = "cuda:8"
                    output_loc = int(device[-1])
                if i == 11:
                    output_loc = int(device[-1])+1
                if i == 12:
                    device = "cuda:9"
                    output_loc = int(device[-1])
                layers.append(block(self.inplanes, planes, output_gpu=output_loc).cuda(device))
        elif self.gpu_num == 5:
            device = "cuda:3"
            layers.append(block(self.inplanes, planes, stride, downsample, output_gpu=3).cuda(device))
            self.inplanes = planes * block.expansion
            output_loc = int(device[-1])
            for i in range(1, blocks):
                if i == 5:
                    output_loc = int(device[-1])+1
                if i == 6:
                    device = "cuda:4"
                layers.append(block(self.inplanes, planes, output_gpu=output_loc).cuda(device))
        elif self.gpu_num == 2:
            device = "cuda:1"
            layers.append(block(self.inplanes, planes, stride, downsample, output_gpu=1).cuda(device))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, output_gpu=1).cuda(device))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        return layers


    def forward(self, x):
        if self.gpu_num == 2:
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            x = self.conv1(s_next)
            x = self.bn1(x)
            x = self.relu(x)
            s_prev = self.layer1(x).cuda('cuda:1')
            ret = []

            for s_next in splits:
                s_prev = self.layer2(s_prev)
                s_prev = self.layer3(s_prev)
                s_prev = self.avgpool(s_prev)
                ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

                x = self.conv1(s_next)
                x = self.bn1(x)
                x = self.relu(x)
                s_prev = self.layer1(x).cuda('cuda:1')
            s_prev = self.layer2(s_prev)
            s_prev = self.layer3(s_prev)
            s_prev = self.avgpool(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
        elif self.gpu_num == 4:
            ## model is put into 4 gpu
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            x = self.conv1(s_next)
            x = self.bn1(x)
            s_prev_1 = self.relu(x).cuda("cuda:1")
            ret = []
            for s_next in splits:
                s_prev_1 = self.layer1(s_prev_1).cuda("cuda:2")
                s_prev_2 = self.layer2(s_prev_1).cuda("cuda:3")
                s_prev_3 = self.layer3(s_prev_2)
                s_prev = self.avgpool(s_prev_3)
                ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

                x = self.conv1(s_next)
                x = self.bn1(x)
                s_prev_1 = self.relu(x).cuda("cuda:1")
            s_prev_1 = self.layer1(s_prev_1).cuda("cuda:2")
            s_prev_2 = self.layer2(s_prev_1).cuda("cuda:3")
            s_prev_3 = self.layer3(s_prev_2)
            s_prev = self.avgpool(s_prev_3)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
        elif self.gpu_num == 5:
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            x = self.conv1(s_next)
            x = self.bn1(x)
            s_prev_1 = self.relu(x).cuda("cuda:1")
            ret = []
            for s_next in splits:
                s_prev_1 = self.layer1(s_prev_1)
                s_prev_2 = self.layer2(s_prev_1)
                s_prev_3 = self.layer3(s_prev_2)
                s_prev = self.avgpool(s_prev_3)
                ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

                x = self.conv1(s_next)
                x = self.bn1(x)
                s_prev_1 = self.relu(x).cuda("cuda:1")
            s_prev_1 = self.layer1(s_prev_1)
            s_prev_2 = self.layer2(s_prev_1)
            s_prev_3 = self.layer3(s_prev_2)
            s_prev = self.avgpool(s_prev_3)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
        elif self.gpu_num == 10:
            ## model is put into 4 gpu
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            x = self.conv1(s_next)
            x = self.bn1(x)
            s_prev_1 = self.relu(x).cuda("cuda:1")
            ret = []
            for s_next in splits:
                s_prev_1 = self.layer1(s_prev_1)
                s_prev_2 = self.layer2(s_prev_1)
                s_prev_3 = self.layer3(s_prev_2)
                s_prev = self.avgpool(s_prev_3)
                ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

                x = self.conv1(s_next)
                x = self.bn1(x)
                s_prev_1 = self.relu(x).cuda("cuda:1")
            s_prev_1 = self.layer1(s_prev_1)
            s_prev_2 = self.layer2(s_prev_1)
            s_prev_3 = self.layer3(s_prev_2)
            s_prev = self.avgpool(s_prev_3)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
        elif self.split_size == 0:
            #print("We are using the correct forward method!!!!!")
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)    # 32x32

            x = self.layer1(x).cuda("cuda:1")  # 32x32
            x = self.layer2(x)  # 16x16
            x = self.layer3(x)  # 8x8

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            x = self.conv1(s_next)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            s_prev = self.layer2(x)
            ret = []

            for s_next in splits:
                s_prev = self.layer3(s_prev)
                s_prev = self.avgpool(s_prev)
                ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

                x = self.conv1(s_next)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.layer1(x)
                s_prev = self.layer2(x)

            s_prev = self.layer3(s_prev)
            s_prev = self.avgpool(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
        return torch.cat(ret)


def resnet_para(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

def resnet164_para(num_classes=10, depth=164, gpu_num = 2, split_size=64, block_name='bottleneck'):
    """
    Constructs a ResNet-164 model.
    """
    return ResNet(depth=164, num_classes=num_classes, block_name=block_name, split_size=split_size, gpu_num =gpu_num)
