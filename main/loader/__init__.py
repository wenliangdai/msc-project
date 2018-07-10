from main.loader.pascal_voc_loader import *
from main.loader.semseg_loader import *
from main.loader.parts_loader import *

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'pascal': pascalVOCLoader,
        'sbd': VOC_Loader,
        'parts': VOC_parts
    }[name]