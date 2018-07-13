from main.loader.loaders import *

def get_loader(name):
    return {
        'sbd': SEMSEG_LOADER,
        'parts': PASCAL_PARTS_LOADER,
        'lip': LIP_LOADER
    }[name]
