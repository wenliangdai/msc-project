import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

output_stride_ref = {'32':3, '16':2, '8':1}
sunet64_path = '/home/wenlidai/sunets-reproduce/main/models/pretrained/SUNets/checkpoint_64_2441_residual.pth.tar'

def sunet(kind='64', num_classes=21, output_stride='32', dprob=1e-7):
    if kind == '64':
        return SUNets(in_dim=512, start_planes=64, filters_base=64, num_classes=num_classes, depth=4, output_stride=output_stride, dprob=dprob)
    elif kind == '128':
        return SUNets(in_dim=512, start_planes=64, filters_base=128, num_classes=num_classes, depth=4, output_stride=output_stride, dprob=dprob)
    elif kind == '7128':
        return SUNets(in_dim=512, start_planes=64, filters_base=128, num_classes=num_classes, depth=7, output_stride=output_stride, dprob=dprob)
    else:
        raise ValueError("Argument {kind} should be '64' or '128' or '7128'.")

class Dilated_sunet64(nn.Module):
    def __init__(self, pretrained=False, num_classes=21, ignore_index=-1, weight=None, output_stride='16', momentum_bn=0.01, dprob=1e-7):
        super(Dilated_sunet64, self).__init__()
        self.num_classes = num_classes
        self.momentum_bn = momentum_bn
        sunet64 = sunet('64', num_classes=num_classes, output_stride=output_stride, dprob=dprob)

        if pretrained:
            # load saved state_dict
            pretrained_state_dict = torch.load(sunet64_path)
            partial_state_dict = OrderedDict()
            for i, (k, v) in enumerate(pretrained_state_dict['state_dict'].items()):
                # print(i)
                if i == len(pretrained_state_dict['state_dict'].items()) - 2:
                    break
                name = k[7:] # remove `module.`
                partial_state_dict[name] = v
            
            new_state_dict = sunet64.state_dict()
            new_state_dict.update(partial_state_dict)
            sunet64.load_state_dict(new_state_dict)

        self.features = sunet64._modules['features'] # A Sequential

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = self.momentum_bn

        # De-gridding filters
        self.final = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2, bias=True)), # size 不变
            ('bn1', nn.BatchNorm2d(512, momentum=self.momentum_bn)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=True)),
            ('bn4', nn.BatchNorm2d(512, momentum=self.momentum_bn)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(512, num_classes, kernel_size=1))
        ]))

    def forward(self, x):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.final(x)
        x = F.upsample(input=x, size=x_size[2:], mode='bilinear', align_corners=True)
        return x

class Dilated_sunet64_multi(nn.Module):
    def __init__(self, pretrained=False, num_classes=[21,20], ignore_index=-1, weight=None, output_stride='16', momentum_bn=0.01, dprob=1e-7):
        super(Dilated_sunet64_multi, self).__init__()
        self.num_classes = num_classes
        self.momentum_bn = momentum_bn
        sunet64 = sunet('64', output_stride=output_stride, dprob=dprob)

        if pretrained:
            # load saved state_dict
            pretrained_state_dict = torch.load(sunet64_path)
            partial_state_dict = OrderedDict()
            for i, (k, v) in enumerate(pretrained_state_dict['state_dict'].items()):
                # print(i)
                if i == len(pretrained_state_dict['state_dict'].items()) - 2:
                    break
                name = k[7:] # remove `module.`
                partial_state_dict[name] = v
            
            new_state_dict = sunet64.state_dict()
            new_state_dict.update(partial_state_dict)
            sunet64.load_state_dict(new_state_dict)

        self.features = sunet64._modules['features'] # A Sequential

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = self.momentum_bn

        # De-gridding filters
        self.final1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2, bias=True)), # size 不变
            ('bn1', nn.BatchNorm2d(512, momentum=self.momentum_bn)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=True)),
            ('bn4', nn.BatchNorm2d(512, momentum=self.momentum_bn)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(512, num_classes[0], kernel_size=1))
        ]))

        self.final2 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2, bias=True)), # size 不变
            ('bn1', nn.BatchNorm2d(512, momentum=self.momentum_bn)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=True)),
            ('bn4', nn.BatchNorm2d(512, momentum=self.momentum_bn)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(512, num_classes[1], kernel_size=1))
        ]))

    def forward(self, x, task=0):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        
        # SBD
        if task == 0:
            x = self.final1(x)
            x = F.upsample(input=x, size=x_size[2:], mode='bilinear', align_corners=True)
            return x
        
        # LIP
        if task == 1:
            x = self.final2(x)
            x = F.upsample(input=x, size=x_size[2:], mode='bilinear', align_corners=True)
            return x
        
        # Human (one image has both two masks)
        if task == 2:
            x1 = self.final1(x)
            x1 = F.upsample(input=x1, size=x_size[2:], mode='bilinear', align_corners=True)
            x2 = self.final1(x)
            x2 = F.upsample(input=x2, size=x_size[2:], mode='bilinear', align_corners=True)
            return x1, x2

class SUNets(nn.Module):
    def __init__(self, in_dim, start_planes=16, filters_base=64, num_classes=1000, depth=1, dprob=1e-7, output_stride='32'):
        super(SUNets, self).__init__()
        self.start_planes = start_planes
        self.depth = depth
        
        # 4 blocks
        filter_factors = [1, 1, 1, 1]
        feature_map_sizes = [filters_base * s for s in filter_factors]

        if filters_base == 128 and depth == 4:
            output_features = [512, 1024, 1536, 2048]
        elif filters_base == 128 and depth == 7:
            output_features = [512, 1280, 2048, 2304]
        elif filters_base == 64 and depth == 4:
            output_features = [256, 512, 768, 1024]

        # num_planes 记录当前时刻，数据的 channels 数量（也就是 depth 维度）
        num_planes = start_planes
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=num_planes, kernel_size=7, stride=2, padding=3, bias=True)),
            ('residual1', ResidualBlock(in_planes=num_planes, out_planes=num_planes*2, dprob=dprob, stride=2))
        ]))
        in_dim = in_dim // 4 # 经过 conv1 和 residual1 两层之后，input dimension 缩小到 1/4
        num_planes = num_planes * 2 # 经过 residual1 之后，数据 channels 数量增大一倍

        block_depth = [2, depth, depth, 1] # 4 UNet blocks, each block contains {depth} UNet modules
        nblocks = 2 # Each UNet block contains E1 and E2 sub-blocks
        for i, d in enumerate(block_depth):
            if i == len(block_depth) - 1: nblocks = 1 # UNet+
            for j in range(d):
                block = UNetModule(
                    in_planes=num_planes, 
                    nblocks=nblocks, 
                    filter_size=feature_map_sizes[i], 
                    dprob=dprob, 
                    in_dim=in_dim, 
                    index=1, 
                    max_planes=output_features[i], 
                    atrous=(i - output_stride_ref[output_stride])*2)
                self.features.add_module('unet%d_%d' % (i + 1, j), block)
                num_planes = output_features[i]
            
            # 如果不是第四个 UNet block
            if i != len(block_depth) - 1:
                # 若已经缩小到对应倍数 output_stride_ref[output_stride]，则 average pooling 保持面积不变，否则面积减半
                if i > output_stride_ref[output_stride] - 1:
                    self.features.add_module('avgpool%d' % (i + 1), nn.AvgPool2d(kernel_size=1, stride=1))
                else:
                    self.features.add_module('avgpool%d' % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2))
                    in_dim = in_dim // 2
        
        self.features.add_module('bn2', nn.BatchNorm2d(num_features=num_planes))        
        self.linear = nn.Linear(in_features=num_planes, out_features=num_classes, bias=True)           

    def forward(self, x):
        out = self.features(x)
        out = F.relu(input=out, inplace=False)
        out = F.avg_pool2d(input=out, kernel_size=7)
        out = out.view(out.size(0), -1) # N x C
        out = self.linear(out)
        return out

class UNetModule(nn.Module):
    def __init__(self, in_planes, nblocks, filter_size, dprob, in_dim, index, max_planes, atrous=0):
        # max_planes is the output feature map size
        super(UNetModule, self).__init__()
        
        self.nblocks = nblocks
        self.in_dim = np.array(in_dim, dtype=float)
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        self.upsample = None
        
        if in_planes != max_planes:
            self.bn = nn.Sequential(OrderedDict([
                ('bn0', nn.BatchNorm2d(in_planes)),
                ('relu0', nn.ReLU(inplace=True))
            ]))
            self.upsample = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_planes, max_planes, kernel_size=1, stride=1, bias=True))
            ]))
        
        for i in range(nblocks): # nblocks is 2 or 1
            if i == 0:
                in_ = in_planes
            else:
                in_ = filter_size

            self.down.append(UNetConv(in_, filter_size, dprob, index and (i == 0), in_planes == max_planes, (2**i)*atrous))
            
            if i == nblocks - 1:
                out_ = filter_size
            else:
                out_ = 2 * filter_size
            
            this_output_padding = 1 - int(np.mod(self.in_dim, 2))
            self.up.append(UNetDeConv(out_, filter_size, dprob, index and (i == 0), max_planes, (2**i)*atrous, output_padding=this_output_padding))
            
            self.in_dim = self.in_dim // 2

    def forward(self, x):
        xs = []
        if self.upsample is not None:
            x = self.bn(x)
        xs.append(x)
        for i, down in enumerate(self.down):
            xout = down(xs[-1])
            xs.append(xout)

        out = xs[-1]
        for i, (x_skip, up) in reversed(list(enumerate(zip(xs[:-1], self.up)))):
            out = up(out)
            if i:
                out = torch.cat([out, x_skip], 1)
            else:
                if self.upsample is not None:
                    x_skip = self.upsample(x_skip)
                out += x_skip
        return out

class UNetConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob, mod_in_planes, is_input_bn, dilation):
        super(UNetConv, self).__init__()
        if mod_in_planes:
            if is_input_bn:
                self.add_module('bn0', nn.BatchNorm2d(in_planes))
                self.add_module('relu0', nn.ReLU(inplace=True))
            self.add_module('conv0', nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True))
            self.add_module('dropout0', nn.Dropout(p=dprob))
            in_planes = out_planes

        self.add_module('bn1', nn.BatchNorm2d(num_features=in_planes))
        self.add_module('relu1', nn.ReLU(inplace=True))

        if dilation > 1:
            # 保持 feature map 面积大小不变
            self.add_module('conv1', nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=True))
        else:
            # W' = (W + 1)/2，面积缩小一半
            self.add_module('conv1', nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=2, bias=True))
        
        self.add_module('dropout1', nn.Dropout(p=dprob))
        self.add_module('bn2', nn.BatchNorm2d(out_planes))
        self.add_module('relu2', nn.ReLU(inplace=True))
        
        if dilation > 1:
            # 保持 feature map 面积大小不变
            self.add_module('conv2', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=2*dilation, stride=1, dilation=2*dilation, bias=True))
        else:
            # 保持 feature map 面积大小不变
            self.add_module('conv2', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True))
        
        self.add_module('dropout2', nn.Dropout(p=dprob))

class UNetDeConv(nn.Sequential):
    # 如果 dilation <= 1，则 output 面积增大一倍
    # 如果 dilation > 1，则 output 面积不变
    def __init__(self, in_planes, out_planes, dprob, mod_in_planes, max_planes, dilation, output_padding=1):
        super(UNetDeConv, self).__init__()
        
        self.add_module('bn0', nn.BatchNorm2d(in_planes))
        self.add_module('relu0', nn.ReLU(inplace=True))
        
        if dilation > 1:
            # 保持 feature map 面积大小不变
            self.add_module('deconv0', nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=1, padding=2 * dilation,
                                                         dilation=2 * dilation, bias=True))
        else:
            # 增大 feature map 面积一倍
            self.add_module('deconv0', nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1,
                                                         output_padding=output_padding, bias=True))
        
        self.add_module('dropout0', nn.Dropout(p=dprob))
        self.add_module('bn1', nn.BatchNorm2d(out_planes))
        self.add_module('relu1', nn.ReLU(inplace=True))
        
        if dilation > 1:
            # 保持 feature map 面积大小不变
            self.add_module('deconv1', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=True))
        else:
            # 保持 feature map 面积大小不变
            self.add_module('deconv1', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True))
        
        self.add_module('dropout1', nn.Dropout(p=dprob))
        
        if mod_in_planes:
            self.add_module('bn2', nn.BatchNorm2d(out_planes))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(out_planes, max_planes, kernel_size=1, bias=True))
            self.add_module('dropout2', nn.Dropout(p=dprob))

class ResidualBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob, stride=1):
        super(ResidualBlock, self).__init__()
        self.bn = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=True),
            nn.Dropout(p=dprob),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True),
            nn.Dropout(p=dprob))
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,out_planes, kernel_size=1, stride=stride, bias=True))
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.bn(x)
        residual = out
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out = self.conv(out)
        out += residual # take the sum
        return out

if __name__ == '__main__':
    # m = sunet('64', num_classes=21, output_stride='8')
    # print(m._modules['features'])

    m = Dilated_sunet64(output_stride='8')
    print(m)