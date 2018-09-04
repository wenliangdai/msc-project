from torch.nn import init

from main.models.sunets import *
from main.models.fcn32 import *

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias.data is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if m.bias.data is not None:
                init.constant_(m.bias, 0)

def get_model(name, n_classes, ignore_index=-1, weight=None, output_stride='16', pretrained=False, momentum_bn=0.01, dprob=0):
    if name in ['sunet64', 'sunet64_multi']:
        model = _get_model_instance(name)
        model = model(
            num_classes=n_classes, 
            ignore_index=ignore_index, 
            weight=weight,
            output_stride=output_stride,
            pretrained=pretrained,
            momentum_bn=momentum_bn,
            dprob=dprob)
        if not pretrained:
            init_params(model.features)
        if name == 'sunet64_multi':
            init_params(model.final1)
            init_params(model.final2)
        else:
            init_params(model.final)
    elif name in ['fcn32vgg']:
        model = _get_model_instance(name)
        model = model(
            num_classes=n_classes,
            pretrained=pretrained)
    elif 'fcn32resnet' in name:
        if 'multi' in name:
            model = _get_model_instance('fcn32resnet_multi')
            depth = int(name.split('_')[0].split('resnet')[-1])
        else:
            model = _get_model_instance('fcn32resnet')
            depth = int(name.split('resnet')[-1])
        model = model(
            num_classes=n_classes,
            pretrained=pretrained,
            depth=depth,
            dprob=dprob)
    else:
        raise 'Model {} not available'.format(name)
    return model

def _get_model_instance(name):
    return {
        'sunet64': Dilated_sunet64,
        'sunet64_multi': Dilated_sunet64_multi,
        'fcn32vgg': FCN32VGG,
        'fcn32vgg_multi': FCN32VGG_MULTI,
        'fcn32resnet': FCN32RESNET,
        'fcn32resnet_multi': FCN32RESNET_MULTI,
        'fcn32alexnet': FCN32ALEXNET
    }[name]
