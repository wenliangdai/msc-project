from main.loader.loaders import *

def get_loader(name):
    return {
        'sbd': SEMSEG_LOADER,
        'parts': PASCAL_PARTS_LOADER,
        'lip': LIP_LOADER,
        'human': PASCAL_HUMAN_LOADER,
        'sbd_lip': SBD_LIP_LOADER
    }[name]
