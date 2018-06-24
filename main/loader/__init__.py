from main.loader.pascal_voc_loader import *
from main.loader.loader import *

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'pascal': pascalVOCLoader,
        'sbd': VOC
    }[name]