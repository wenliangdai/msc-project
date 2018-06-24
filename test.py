class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = dotdict({
    'arch': 'sunet64',
    'batch_size': 22,
    'dataset': 'sbd',
    'freeze': False,
    'img_cols': 512,
    'img_rows': 512,
    'iter_size': 1,
    'lr': 0.0002*10,
    'log_size': 400,
    'manual_seed': 0,
    'model_path': None,
    'momentum': 0.95,
    'epochs': 90,
    'optim': 'SGD',
    'output_stride': '16',
    'restore': False,
    'split': 'train_aug',
    'weight_decay': 0.0001
})

def main():
    print(args.arch)


if __name__ == '__main__':
    main()