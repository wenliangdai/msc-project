import argparse
import math
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data

import torchvision.transforms as transforms
import transforms as extended_transforms
from loss import prediction_stat
from main import get_data_path
from main.loader import get_loader
from main.models import get_model
from utils import dotdict, float2str

# paths
ROOT = '/home/wenlidai/msc-project/'
RESULT = 'results'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    print('='*10, 'Starting', '='*10, '\n')
    print(device)

    # Set the seed for reproducing the results
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = True
    
    # Set up results folder
    if not os.path.exists(os.path.join(ROOT, RESULT, 'saved_val_images')):
        os.makedirs(os.path.join(ROOT, RESULT, 'saved_val_images'))
    if not os.path.exists(os.path.join(ROOT, RESULT, 'saved_train_images')):
        os.makedirs(os.path.join(ROOT, RESULT, 'saved_train_images'))

    # Setup Dataloader
    data_loader = get_loader(args.dataset)

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = extended_transforms.MaskToTensor()

    traindata = data_loader(
        'train', 
        n_classes=args.n_classes, 
        transform=input_transform, 
        target_transform=target_transform, 
        do_transform=True, 
        portion=args.data_portion
    )
    trainloader = data.DataLoader(traindata, batch_size=args.batch_size, num_workers=1, shuffle=True)
    valdata = data_loader(
        'val', 
        n_classes=args.n_classes, 
        transform=input_transform, 
        target_transform=target_transform
    )
    valloader = data.DataLoader(valdata, batch_size=args.batch_size, num_workers=1, shuffle=False)

    n_classes = traindata.n_classes
    n_trainsamples = len(traindata)
    n_iters_per_epoch = np.ceil(n_trainsamples / float(args.batch_size * args.iter_size))

    print('#Training data = {}'.format(n_trainsamples))

    # Setup Model
    model = get_model(
        name=args.arch, 
        n_classes=n_classes, 
        ignore_index=traindata.ignore_index, 
        output_stride=args.output_stride,
        pretrained=args.pretrained,
        momentum_bn=args.momentum_bn,
        dprob=args.dprob
    ).to(device)

    epochs_done=0
    X=[]
    Y=[]
    Y_test=[]
    avg_pixel_acc = 0
    mean_class_acc = 0
    mIoU = 0
    avg_pixel_acc_test = 0
    mean_class_acc_test = 0
    mIoU_test = 0
    best_mIoU = 0
    best_epoch = 0

    if args.model_path:
        model_name = args.model_path.split('.')
        checkpoint_name = model_name[0] + '_optimizer.pkl'
        checkpoint = torch.load(os.path.join(ROOT, RESULT, checkpoint_name))
        optm = checkpoint['optimizer']
        model.load_state_dict(checkpoint['state_dict'])
        split_str = model_name[0].split('_')
        epochs_done = int(split_str[-1])
        saved_loss = pickle.load( open(os.path.join(ROOT, RESULT, "saved_loss.p"), "rb") )
        saved_accuracy = pickle.load( open(os.path.join(ROOT, RESULT, "saved_accuracy.p"), "rb") )
        X=saved_loss["X"][:epochs_done]
        Y=saved_loss["Y"][:epochs_done]
        Y_test=saved_loss["Y_test"][:epochs_done]
        avg_pixel_acc = saved_accuracy["P"][:epochs_done,:]
        mean_class_acc = saved_accuracy["M"][:epochs_done,:]
        mIoU = saved_accuracy["I"][:epochs_done,:]
        avg_pixel_acc_test = saved_accuracy["P_test"][:epochs_done,:]
        mean_class_acc_test = saved_accuracy["M_test"][:epochs_done,:]
        mIoU_test = saved_accuracy["I_test"][:epochs_done,:]
    
    if args.best_model_path:
        best_model_name = args.best_model_path.split('_')
        best_mIoU = float(best_model_name[-2])
        best_epoch = int(best_model_name[-3])

    # Learning rates: For new layers (such as final layer), we set lr to be 10x the learning rate of layers already trained
    bias_10x_params = filter(lambda x: ('bias' in x[0]) and ('final' in x[0]) and ('conv' in x[0]),
                         model.named_parameters())
    bias_10x_params = list(map(lambda x: x[1], bias_10x_params))

    bias_params = filter(lambda x: ('bias' in x[0]) and ('final' not in x[0]),
                         model.named_parameters())
    bias_params = list(map(lambda x: x[1], bias_params))

    nonbias_10x_params = filter(lambda x: (('bias' not in x[0]) or ('bn' in x[0])) and ('final' in x[0]),
                         model.named_parameters())
    nonbias_10x_params = list(map(lambda x: x[1], nonbias_10x_params))

    nonbias_params = filter(lambda x: ('bias' not in x[0]) and ('final' not in x[0]),
                            model.named_parameters())
    nonbias_params = list(map(lambda x: x[1], nonbias_params))

    optimizer = torch.optim.SGD([{'params': bias_params, 'lr': args.lr},
                                 {'params': bias_10x_params, 'lr': 20 * args.lr if args.pretrained else args.lr},
                                 {'params': nonbias_10x_params, 'lr': 10 * args.lr if args.pretrained else args.lr},
                                 {'params': nonbias_params, 'lr': args.lr},],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=(args.optim == 'Nesterov'))
    num_param_groups = 4

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Setting up scheduler
    if args.model_path and args.restore:
        # Here we restore all states of optimizer
        optimizer.load_state_dict(optm)
        total_iters = n_iters_per_epoch * args.epochs
        lambda1 = lambda step: 0.5 + 0.5 * math.cos(np.pi * step / total_iters)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1]*num_param_groups, last_epoch=epochs_done*n_iters_per_epoch)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=epochs_done)
    else:
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        # Here we simply restart the training
        # if args.T0:
        #     total_iters = args.T0 * n_iters_per_epoch
        # else:
        total_iters = ((args.epochs - epochs_done) * n_iters_per_epoch)
        lambda1 = lambda step: 0.5 + 0.5 * math.cos(np.pi * step / total_iters)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1]*num_param_groups)

    

    global l_avg, totalclasswise_pixel_acc, totalclasswise_gtpixels, totalclasswise_predpixels
    global l_avg_test, totalclasswise_pixel_acc_test, totalclasswise_gtpixels_test, totalclasswise_predpixels_test
    global steps, steps_test

    scheduler.step()

    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=traindata.ignore_index)

    for epoch in range(epochs_done, args.epochs):
        print('='*10, 'Epoch %d' % (epoch + 1), '='*10)
        l_avg = 0
        totalclasswise_pixel_acc = 0
        totalclasswise_gtpixels = 0
        totalclasswise_predpixels = 0
        l_avg_test = 0
        totalclasswise_pixel_acc_test = 0
        totalclasswise_gtpixels_test = 0
        totalclasswise_predpixels_test = 0
        steps = 0
        steps_test = 0
        
        # scheduler.step()
        train(model, optimizer, criterion, trainloader, epoch, scheduler, traindata)
        val(model, criterion, valloader, epoch, valdata)

        # save the model every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            if (epoch + 1) > 5:
                os.remove(os.path.join(ROOT, RESULT, "{}_{}_{}.pkl".format(args.arch, args.dataset, epoch - 4)))
                os.remove(os.path.join(ROOT, RESULT, "{}_{}_{}_optimizer.pkl".format(args.arch, args.dataset, epoch - 4)))
            torch.save(model, os.path.join(ROOT, RESULT, "{}_{}_{}.pkl".format(args.arch, args.dataset, epoch + 1)))
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(ROOT, RESULT, "{}_{}_{}_optimizer.pkl".format(args.arch, args.dataset, epoch + 1)))
        
        # remove old loss & accuracy files
        if os.path.isfile(os.path.join(ROOT, RESULT, "saved_loss.p")):
            os.remove(os.path.join(ROOT, RESULT, "saved_loss.p"))
        if os.path.isfile(os.path.join(ROOT, RESULT, "saved_accuracy.p")):
            os.remove(os.path.join(ROOT, RESULT, "saved_accuracy.p"))

        # save train and validation loss
        X.append(epoch + 1)
        Y.append(l_avg / steps)
        Y_test.append(l_avg_test / steps_test)
        saved_loss={"X": X, "Y": Y, "Y_test": Y_test}
        pickle.dump(saved_loss, open(os.path.join(ROOT, RESULT, "saved_loss.p"), "wb"))
        
        # pixel accuracy
        totalclasswise_pixel_acc = totalclasswise_pixel_acc.reshape((-1, n_classes)).astype(np.float32)
        totalclasswise_gtpixels = totalclasswise_gtpixels.reshape((-1, n_classes))
        totalclasswise_predpixels = totalclasswise_predpixels.reshape((-1, n_classes))
        totalclasswise_pixel_acc_test = totalclasswise_pixel_acc_test.reshape((-1, n_classes)).astype(np.float32)
        totalclasswise_gtpixels_test = totalclasswise_gtpixels_test.reshape((-1, n_classes))
        totalclasswise_predpixels_test = totalclasswise_predpixels_test.reshape((-1, n_classes))

        if isinstance(avg_pixel_acc, np.ndarray):
            avg_pixel_acc = np.vstack((avg_pixel_acc, np.sum(totalclasswise_pixel_acc, axis=1) / np.sum(totalclasswise_gtpixels, axis=1)))
            mean_class_acc = np.vstack((mean_class_acc, np.mean(totalclasswise_pixel_acc / totalclasswise_gtpixels, axis=1)))
            mIoU = np.vstack((mIoU, np.mean(totalclasswise_pixel_acc / (totalclasswise_gtpixels + totalclasswise_predpixels - totalclasswise_pixel_acc), axis=1)))

            avg_pixel_acc_test = np.vstack((avg_pixel_acc_test, np.sum(totalclasswise_pixel_acc_test,axis=1) / np.sum(totalclasswise_gtpixels_test, axis=1)))
            mean_class_acc_test = np.vstack((mean_class_acc_test, np.mean(totalclasswise_pixel_acc_test / totalclasswise_gtpixels_test, axis=1)))
            mIoU_test = np.vstack((mIoU_test, np.mean(totalclasswise_pixel_acc_test / (totalclasswise_gtpixels_test + totalclasswise_predpixels_test - totalclasswise_pixel_acc_test), axis=1)))
        else:
            avg_pixel_acc = np.sum(totalclasswise_pixel_acc, axis=1) / np.sum(totalclasswise_gtpixels, axis=1)
            mean_class_acc = np.mean(totalclasswise_pixel_acc / totalclasswise_gtpixels, axis=1)
            mIoU = np.mean(totalclasswise_pixel_acc / (totalclasswise_gtpixels + totalclasswise_predpixels - totalclasswise_pixel_acc), axis=1)

            avg_pixel_acc_test = np.sum(totalclasswise_pixel_acc_test, axis=1) / np.sum(totalclasswise_gtpixels_test, axis=1)
            mean_class_acc_test = np.mean(totalclasswise_pixel_acc_test / totalclasswise_gtpixels_test, axis=1)
            mIoU_test = np.mean(totalclasswise_pixel_acc_test / (totalclasswise_gtpixels_test + totalclasswise_predpixels_test - totalclasswise_pixel_acc_test), axis=1)

        saved_accuracy = {"X": X, "P": avg_pixel_acc, "M": mean_class_acc, "I": mIoU,
                          "P_test": avg_pixel_acc_test, "M_test": mean_class_acc_test, "I_test": mIoU_test}
        pickle.dump(saved_accuracy, open(os.path.join(ROOT, RESULT, "saved_accuracy.p"), "wb"))

        # save the best model
        this_mIoU = np.mean(totalclasswise_pixel_acc_test / (totalclasswise_gtpixels_test + totalclasswise_predpixels_test - totalclasswise_pixel_acc_test), axis=1)[0]
        if this_mIoU > best_mIoU:
            if best_mIoU > 0:
                os.remove(os.path.join(ROOT, RESULT, "{}_{}_{}_{}_best.pkl".format(args.arch, args.dataset, best_epoch, float2str(best_mIoU))))
                os.remove(os.path.join(ROOT, RESULT, "{}_{}_{}_{}_optimizer_best.pkl".format(args.arch, args.dataset, best_epoch, float2str(best_mIoU))))
            best_mIoU = this_mIoU
            best_epoch = epoch + 1
            torch.save(model, os.path.join(ROOT, RESULT, "{}_{}_{}_{}_best.pkl".format(args.arch, args.dataset, best_epoch, float2str(best_mIoU))))
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(ROOT, RESULT, "{}_{}_{}_{}_optimizer_best.pkl".format(args.arch, args.dataset, best_epoch, float2str(best_mIoU))))

        train_mIoU = np.mean(totalclasswise_pixel_acc / (totalclasswise_gtpixels + totalclasswise_predpixels - totalclasswise_pixel_acc), axis=1)[0]
        print('Train mIoU = {}, Val mIoU = {}'.format(train_mIoU, this_mIoU))

# Incase one want to freeze BN params
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def train(model, optimizer, criterion, trainloader, epoch, scheduler, data):
    global l_avg, totalclasswise_pixel_acc, totalclasswise_gtpixels, totalclasswise_predpixels
    global steps

    model.train()

    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_valid_pixel = float((labels.data != criterion.ignore_index).long().sum())

        total_loss = loss.sum()
        total_loss = total_loss / float(total_valid_pixel)
        total_loss.backward()

        classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([outputs], labels, data.n_classes)
        classwise_pixel_acc = torch.FloatTensor(classwise_pixel_acc)
        classwise_gtpixels = torch.FloatTensor(classwise_gtpixels)
        classwise_predpixels = torch.FloatTensor(classwise_predpixels)

        l_avg += loss.sum().data.cpu().numpy()
        steps += total_valid_pixel
        totalclasswise_pixel_acc += classwise_pixel_acc.numpy()
        totalclasswise_gtpixels += classwise_gtpixels.numpy()
        totalclasswise_predpixels += classwise_predpixels.numpy()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # if (i + 1) % args.log_size == 0:
        #     pickle.dump(images[0].cpu().numpy(),
        #                 open(os.path.join(ROOT, RESULT, "saved_train_images/" + str(epoch) + "_" + str(i) + "_input.p"), "wb"))
        #     pickle.dump(np.transpose(data.decode_segmap(outputs[0].data.cpu().numpy().argmax(0)), [2, 0, 1]),
        #                 open(os.path.join(ROOT, RESULT, "saved_train_images/" + str(epoch) + "_" + str(i) + "_output.p"), "wb"))
        #     pickle.dump(np.transpose(data.decode_segmap(labels[0].cpu().numpy()), [2, 0, 1]),
        #                 open(os.path.join(ROOT, RESULT, "saved_train_images/" + str(epoch) + "_" + str(i) + "_target.p"), "wb"))

def val(model, criterion, valloader, epoch, data):
    global l_avg_test, totalclasswise_pixel_acc_test, totalclasswise_gtpixels_test, totalclasswise_predpixels_test
    global steps_test

    model.eval()

    for i, (images, labels) in enumerate(valloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_valid_pixel = total_valid_pixel = float((labels.data != criterion.ignore_index).long().sum())
            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([outputs], labels, data.n_classes)
            classwise_pixel_acc = torch.FloatTensor(classwise_pixel_acc)
            classwise_gtpixels = torch.FloatTensor(classwise_gtpixels)
            classwise_predpixels = torch.FloatTensor(classwise_predpixels)

            l_avg_test += loss.sum().data.cpu().numpy()
            steps_test += total_valid_pixel
            totalclasswise_pixel_acc_test += classwise_pixel_acc.numpy()
            totalclasswise_gtpixels_test += classwise_gtpixels.numpy()
            totalclasswise_predpixels_test += classwise_predpixels.numpy()

            # if (i + 1) % 100 == 0:
            #     pickle.dump(images[0].cpu().numpy(),
            #                 open(os.path.join(ROOT, RESULT, "saved_val_images/" + str(epoch) + "_" + str(i) + "_input.p"), "wb"))
            #     pickle.dump(np.transpose(data.decode_segmap(outputs[0].data.cpu().numpy().argmax(0)), [2, 0, 1]),
            #                 open(os.path.join(ROOT, RESULT, "saved_val_images/" + str(epoch) + "_" + str(i) + "_output.p"), "wb"))
            #     pickle.dump(np.transpose(data.decode_segmap(labels[0].cpu().numpy()), [2, 0, 1]),
            #                 open(os.path.join(ROOT, RESULT, "saved_val_images/" + str(epoch) + "_" + str(i) + "_target.p"), "wb"))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='sunet64',
                        help='Architecture to use [\'sunet64, sunet128, sunet7128 etc\']')
    parser.add_argument('--model_path', help='Path to the saved model', type=str)
    parser.add_argument('--best_model_path', help='Path to the saved best model', type=str)
    parser.add_argument('--dataset', nargs='?', type=str, default='sbd',
                        help='Dataset to use [\'sbd, coco, cityscapes etc\']')
    parser.add_argument('--data_portion', nargs='?', type=float, default=1.0,
                        help='Portion of dataset to use')
    parser.add_argument('--img_rows', nargs='?', type=int, default=512,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=512,
                        help='Width of the input image')
    parser.add_argument('--epochs', nargs='?', type=int, default=90,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10,
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=0.0005,
                        help='Learning Rate')
    parser.add_argument('--manual_seed', default=0, type=int,
                        help='manual seed')
    parser.add_argument('--iter_size', type=int, default=1,
                        help='number of batches per weight updates')
    parser.add_argument('--log_size', type=int, default=400,
                        help='iteration period of logging segmented images')
    parser.add_argument('--dprob', nargs='?', type=float, default=0,
                        help='Dropout probability')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.95,
                        help='Momentum for SGD')
    parser.add_argument('--momentum_bn', nargs='?', type=float, default=0.01,
                        help='Momentum for BN')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--output_stride', nargs='?', type=str, default='16',
                        help='Output stride to use [\'32, 16, 8 etc\']')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze BN params')
    parser.add_argument('--restore', action='store_true',
                        help='Restore Optimizer params')
    parser.add_argument('--epoch_log_size', nargs='?', type=str, default=20,
                        help='Every [epoch_log_size] iterations to print loss in each epoch')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained ImageNet initialization or not')
    parser.add_argument('--n_classes', nargs='?', type=int, default=21,
                        help='number of classes of the labels')
    parser.add_argument('--optim', nargs='?', type=str, default='SGD',
                        help='Optimizer to use [\'SGD, Nesterov etc\']')

    global args
    args = parser.parse_args()
    
    RESULT = '{}_{}_{}_{}'.format(RESULT, args.arch, args.dataset, args.weight_decay)
    if args.pretrained:
        RESULT = RESULT + '_pretrained'
    
    main(args)


# args = dotdict({
#     'arch': 'sunet64',
#     'batch_size': 10,
#     'dataset': 'sbd',
#     'freeze': False,
#     'img_cols': 512,
#     'img_rows': 512,
#     'iter_size': 1,
#     'lr': 0.0005,
#     'log_size': 800,
#     'epoch_log_size': 20,
#     'manual_seed': 0,
#     'model_path': 'sunet64_sbd_100.pkl',
#     'best_model_path': 'sunet64_sbd_102_0.31_best.pkl',
#     'momentum': 0.90,
#     'epochs': 120,
#     'optim': 'SGD',
#     'output_stride': '16',
#     'restore': True,
#     'split': 'train_aug',
#     'weight_decay': 0.0005
# })
