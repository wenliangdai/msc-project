from main.loader.pascal_voc_loader import *
from main.loader.semseg_loader import *
from main.loader.parts_loader import *
from main.loader.lip_loader import *

def get_loader(name):
    return {
        'sbd': VOC_Loader,
        'parts': PASCAL_PARTS_LOADER,
        'lip': LIP_LOADER
    }[name]
