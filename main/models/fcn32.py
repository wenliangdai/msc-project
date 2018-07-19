import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models
from utils import get_upsampling_weight

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

        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, 
            nn.ReLU(inplace=True), 
            nn.Dropout(), 
            fc7, 
            nn.ReLU(inplace=True), 
            nn.Dropout(), 
            score_fr
        )

        # Upsampling
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 64))
    
        # print(vgg)
    
    def forward(self, x):
        this_shape = x.size()
        conv_out = self.features5(x)
        score_out = self.score_fr(conv_out)
        upsampling_out = self.upscore(score_out)
        # .contiguous() makes sure the tensor is stored in a contiguous chunk of memory
        # return upsampling_out[:, :, 19:(19 + this_shape[2]), 19:(19 + this_shape[3])].contiguous()
        upsampling_out = F.upsample(input=upsampling_out, size=this_shape[2:], mode='bilinear', align_corners=True)
        return upsampling_out
