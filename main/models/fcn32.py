import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models
from utils import get_upsampling_weight
from collections import OrderedDict

class FCN32VGG(nn.Module):
    def __init__(self, num_classes=21, pretrained=False):
        super(FCN32VGG, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        
        # Why pad the input: 
        # https://github.com/shelhamer/fcn.berkeleyvision.org#frequently-asked-questions
        # features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        # As the shapes are different, we can't use load_state_dict/state_dict directly
        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        param6 = classifier[0].state_dict()
        param6['weight'] = param6['weight'].view(4096, 512, 7, 7)
        fc6.load_state_dict(param6, strict=True)

        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        param7 = classifier[3].state_dict()
        param7['weight'] = param7['weight'].view(4096, 4096, 1, 1)
        fc7.load_state_dict(param7, strict=True)

        final = nn.Conv2d(4096, num_classes, kernel_size=1)
        final.weight.data.zero_()
        final.bias.data.zero_()
        upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        upscore.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 64))
        self.final = nn.Sequential(OrderedDict([
            ('conv0', fc6), 
            ('relu1', nn.ReLU(inplace=True)), 
            ('dropout2', nn.Dropout()), 
            ('conv3', fc7), 
            ('relu4', nn.ReLU(inplace=True)), 
            ('dropout5', nn.Dropout()), 
            ('conv6', final),
            ('tconv7', upscore)
        ]))
    
    def forward(self, x):
        this_shape = x.size()
        x = self.features5(x)
        x = self.final(x)
        x = F.upsample(input=x, size=this_shape[2:], mode='bilinear', align_corners=True)
        return x


class FCN32VGG_MULTI(nn.Module):
    def __init__(self, num_classes=[21,20], pretrained=False):
        super(FCN32VGG_MULTI, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        
        # Why pad the input: 
        # https://github.com/shelhamer/fcn.berkeleyvision.org#frequently-asked-questions
        # features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        # As the shapes are different, we can't use load_state_dict/state_dict directly
        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        param6 = classifier[0].state_dict()
        param6['weight'] = param6['weight'].view(4096, 512, 7, 7)
        fc6.load_state_dict(param6, strict=True)

        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        param7 = classifier[3].state_dict()
        param7['weight'] = param7['weight'].view(4096, 4096, 1, 1)
        fc7.load_state_dict(param7, strict=True)

        final1 = nn.Conv2d(4096, num_classes[0], kernel_size=1)
        final1.weight.data.zero_()
        final1.bias.data.zero_()
        upscore1 = nn.ConvTranspose2d(num_classes[0], num_classes[0], kernel_size=64, stride=32, bias=False)
        upscore1.weight.data.copy_(get_upsampling_weight(num_classes[0], num_classes[0], 64))
        self.final1 = nn.Sequential(OrderedDict([
            ('conv0', fc6), 
            ('relu1', nn.ReLU(inplace=True)), 
            ('dropout2', nn.Dropout()), 
            ('conv3', fc7), 
            ('relu4', nn.ReLU(inplace=True)), 
            ('dropout5', nn.Dropout()), 
            ('conv6', final1),
            ('tconv7', upscore1)
        ]))

        final2 = nn.Conv2d(4096, num_classes[1], kernel_size=1)
        final2.weight.data.zero_()
        final2.bias.data.zero_()
        upscore2 = nn.ConvTranspose2d(num_classes[1], num_classes[1], kernel_size=64, stride=32, bias=False)
        upscore2.weight.data.copy_(get_upsampling_weight(num_classes[1], num_classes[1], 64))
        self.final2 = nn.Sequential(OrderedDict([
            ('conv0', fc6), 
            ('relu1', nn.ReLU(inplace=True)), 
            ('dropout2', nn.Dropout()), 
            ('conv3', fc7), 
            ('relu4', nn.ReLU(inplace=True)), 
            ('dropout5', nn.Dropout()), 
            ('conv6', final2),
            ('tconv7', upscore2)
        ]))
    
    def forward(self, x, task):
        this_shape = x.size()
        x = self.features5(x)
        x = self.final1(x) if task == 0 else self.final2(x)
        x = F.upsample(input=x, size=this_shape[2:], mode='bilinear', align_corners=True)
        return x


class FCN32RESNET(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, depth=18, dprob=0.1):
        super(FCN32RESNET, self).__init__()
        print('pretrained = {}, depth = {}'.format(pretrained, depth))
        if depth == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif depth == 34:
            resnet = models.resnet34(pretrained=pretrained)
        elif depth == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif depth == 101:
            resnet = models.resnet101(pretrained=pretrained)
        elif depth == 152:
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise TypeError('Invalid Resnet depth')

        features = [*resnet.children()]
        num_channels = features[-1].in_features
        features = features[0:-1] # remove the original 1000-dimension Linear layer

        for f in features:
            if 'MaxPool' in f.__class__.__name__ or 'AvgPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        # Add Dropout module after each conv layer for torchvision.models.resnet
        modified_features = []
        for f in features:
            if f.__class__.__name__ == 'Sequential':
                new_seq = []
                for ff in f.children():
                    list_modules = [*ff.children()]
                    for module in list_modules:
                        new_seq.append(module)
                        if 'Conv' in module.__class__.__name__:
                            new_seq.append(nn.Dropout(p=dprob))
                modified_features.append(nn.Sequential(*new_seq))
            else:
                modified_features.append(f)

        self.features = nn.Sequential(*modified_features)

        final = nn.Conv2d(num_channels, num_classes, kernel_size=1)
        final.weight.data.zero_()
        final.bias.data.zero_()
        upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        upscore.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 64))
        self.final = nn.Sequential(OrderedDict([
            ('conv6', final),
            ('tconv7', upscore)
        ]))
    
    def forward(self, x):
        this_shape = x.size()
        x = self.features(x)
        x = self.final(x)
        x = F.upsample(input=x, size=this_shape[2:], mode='bilinear', align_corners=True)
        return x

class FCN32RESNET_MULTI(nn.Module):
    def __init__(self, num_classes=[21,20], pretrained=False, depth=18):
        super(FCN32RESNET_MULTI, self).__init__()
        if depth == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif depth == 34:
            resnet = models.resnet34(pretrained=pretrained)
        elif depth == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif depth == 101:
            resnet = models.resnet101(pretrained=pretrained)
        elif depth == 152:
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise TypeError('Invalid Resnet depth')

        features = [*resnet.children()]
        num_channels = features[-1].in_features
        features = features[0:-1] # remove the original 1000-dimension Linear layer

        for f in features:
            if 'MaxPool' in f.__class__.__name__ or 'AvgPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features = nn.Sequential(*features)

        final1 = nn.Conv2d(num_channels, num_classes[0], kernel_size=1)
        final1.weight.data.zero_()
        final1.bias.data.zero_()
        upscore1 = nn.ConvTranspose2d(num_classes[0], num_classes[0], kernel_size=64, stride=32, bias=False)
        upscore1.weight.data.copy_(get_upsampling_weight(num_classes[0], num_classes[0], 64))
        self.final1 = nn.Sequential(OrderedDict([
            ('conv6', final1),
            ('tconv7', upscore1)
        ]))

        final2 = nn.Conv2d(num_channels, num_classes[1], kernel_size=1)
        final2.weight.data.zero_()
        final2.bias.data.zero_()
        upscore2 = nn.ConvTranspose2d(num_classes[1], num_classes[1], kernel_size=64, stride=32, bias=False)
        upscore2.weight.data.copy_(get_upsampling_weight(num_classes[1], num_classes[1], 64))
        self.final2 = nn.Sequential(OrderedDict([
            ('conv6', final2),
            ('tconv7', upscore2)
        ]))
    
    def forward(self, x, task):
        this_shape = x.size()
        x = self.features(x)
        x = self.final1(x) if task == 0 else self.final2(x)
        x = F.upsample(input=x, size=this_shape[2:], mode='bilinear', align_corners=True)
        return x