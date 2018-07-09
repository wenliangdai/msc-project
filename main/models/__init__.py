from torch.nn import init

from main.models.sunets import *

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

def get_model(name, n_classes, ignore_index=-1, weight=None, output_stride='16', pretrained=False):
    if name == 'sunet64':
        model = _get_model_instance(name)
        model = model(
            num_classes=n_classes, 
            ignore_index=ignore_index, 
            weight=weight,
            output_stride=output_stride,
            pretrained=pretrained)
        if not pretrained:
            init_params(model.features)
        init_params(model.final)
    else:
        raise 'Model {} not available'.format(name)
    return model

def _get_model_instance(name):
    return {
        'sunet64': Dilated_sunet64
    }[name]
