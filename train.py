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

import torchvision.transforms as standard_transforms
import transforms as extended_transforms
from loss import prediction_stat, prediction_stat_confusion_matrix
from main import get_data_path
from main.loader import get_loader
from main.models import get_model
from utils import dotdict

ROOT_ADDRESS = '/home/wenlidai/sunets-reproduce/'

torch.set_printoptions(threshold=1e5)

args = dotdict({
    'arch': 'sunet64',
    'batch_size': 10,
    'dataset': 'sbd',
    'freeze': False,
    'img_cols': 512,
    'img_rows': 512,
    'iter_size': 1,
    'lr': 0.0002*10,
    'log_size': 400,
    'epoch_log_size': 20,
    'manual_seed': 0,
    'model_path': os.path.join(ROOT_ADDRESS, 'results', 'sunet64_sbd_90.pkl'),
    'momentum': 0.95,
    'epochs': 150,
    'optim': 'SGD',
    'output_stride': '16',
    'restore': True,
    'split': 'train_aug',
    'weight_decay': 0.0001
})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    print('='*10, 'Starting', '='*10, '\n')

    # Set the seed for reproducing the results
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = True
    
    # Set up results folder
    if not os.path.exists(os.path.join(ROOT_ADDRESS, 'results/saved_val_images')):
        os.makedirs(os.path.join(ROOT_ADDRESS, 'results/saved_val_images'))
    if not os.path.exists(os.path.join(ROOT_ADDRESS, 'results/saved_train_images')):
        os.makedirs(os.path.join(ROOT_ADDRESS, 'results/saved_train_images'))

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)

    # traindata = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols))
    # trainloader = data.DataLoader(traindata, batch_size=args.batch_size, num_workers=7, shuffle=True)
    # valdata = data_loader(data_path, split="val", is_transform=False, img_size=(args.img_rows, args.img_cols))
    # valloader = data.DataLoader(valdata, batch_size=args.batch_size, num_workers=7, shuffle=False)

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()

    traindata = data_loader('train', transform=input_transform, target_transform=target_transform, do_transform=True)
    trainloader = data.DataLoader(traindata, batch_size=args.batch_size, num_workers=1, shuffle=True)
    valdata = data_loader('train', transform=input_transform, target_transform=target_transform)
    valloader = data.DataLoader(valdata, batch_size=args.batch_size, num_workers=1, shuffle=False)

    n_classes = traindata.n_classes
    n_trainsamples = len(traindata)
    n_iters_per_epoch = np.ceil(n_trainsamples / float(args.batch_size * args.iter_size))

    # Setup Model
    model = get_model(args.arch, n_classes, ignore_index=traindata.ignore_index, output_stride=args.output_stride).to(device)

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

    if args.model_path:
        model_name = args.model_path.split('.')
        checkpoint_name = model_name[0] + '_optimizer.pkl'
        checkpoint = torch.load(checkpoint_name)
        optm = checkpoint['optimizer']
        model.load_state_dict(checkpoint['state_dict'])
        split_str = model_name[0].split('_')
        epochs_done = int(split_str[-1])
        saved_loss = pickle.load( open(os.path.join(ROOT_ADDRESS, "results/saved_loss.p"), "rb") )
        saved_accuracy = pickle.load( open(os.path.join(ROOT_ADDRESS, "results/saved_accuracy.p"), "rb") )
        X=saved_loss["X"][:epochs_done]
        Y=saved_loss["Y"][:epochs_done]
        Y_test=saved_loss["Y_test"][:epochs_done]
        avg_pixel_acc = saved_accuracy["P"][:epochs_done,:]
        mean_class_acc = saved_accuracy["M"][:epochs_done,:]
        mIoU = saved_accuracy["I"][:epochs_done,:]
        avg_pixel_acc_test = saved_accuracy["P_test"][:epochs_done,:]
        mean_class_acc_test = saved_accuracy["M_test"][:epochs_done,:]
        mIoU_test = saved_accuracy["I_test"][:epochs_done,:]

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
                                 {'params': bias_10x_params, 'lr': args.lr},
                                 {'params': nonbias_10x_params, 'lr': args.lr},
                                 {'params': nonbias_params, 'lr': args.lr},],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=(args.optim == 'Nesterov'))
    num_param_groups = 4

    # Setting up scheduler
    lambda1 = lambda step: 0.5 + 0.5 * math.cos(np.pi * step / total_iters)
    if args.model_path and args.restore:
        # Here we restore all states of optimizer
        optimizer.load_state_dict(optm)
        total_iters = n_iters_per_epoch * args.epochs
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1]*num_param_groups, last_epoch=epochs_done*n_iters_per_epoch)
    else:
        # Here we simply restart the training
        if args.T0:
            total_iters = args.T0 * n_iters_per_epoch
        else:
            total_iters = ((args.epochs - epochs_done) * n_iters_per_epoch)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1]*num_param_groups)

    global l_avg, totalclasswise_pixel_acc, totalclasswise_gtpixels, totalclasswise_predpixels
    global l_avg_test, totalclasswise_pixel_acc_test, totalclasswise_gtpixels_test, totalclasswise_predpixels_test
    global steps, steps_test

    scheduler.step()

    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=traindata.ignore_index)

    print('='*10, 'Entering epoch loop', '='*10, '\n')
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
        
        train(model, optimizer, criterion, trainloader, epoch, scheduler, traindata)
        val(model, criterion, valloader, epoch, valdata)

        # save the model every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model, os.path.join(ROOT_ADDRESS, "results/{}_{}_{}.pkl".format(args.arch, args.dataset, epoch + 1)))
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(ROOT_ADDRESS, "results/{}_{}_{}_optimizer.pkl".format(args.arch, args.dataset, epoch + 1)))
        
        # remove old loss & accuracy files
        if os.path.isfile(os.path.join(ROOT_ADDRESS, "results/saved_loss.p")):
            os.remove(os.path.join(ROOT_ADDRESS, "results/saved_loss.p"))
        if os.path.isfile(os.path.join(ROOT_ADDRESS, "results/saved_accuracy.p")):
            os.remove(os.path.join(ROOT_ADDRESS, "results/saved_accuracy.p"))

        # saving train and validation loss
        X.append(epoch + 1)
        Y.append(l_avg / steps)
        Y_test.append(l_avg_test / steps_test)
        saved_loss={"X": X, "Y": Y, "Y_test": Y_test}
        pickle.dump(saved_loss, open(os.path.join(ROOT_ADDRESS, "results/saved_loss.p"), "wb"))
        
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
        pickle.dump(saved_accuracy, open(os.path.join(ROOT_ADDRESS, "results/saved_accuracy.p"), "wb"))


# Incase one want to freeze BN params
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

# def transform_labels(labels):


def train(model, optimizer, criterion, trainloader, epoch, scheduler, data):
    print('='*10, 'Train step', '='*10, '\n')

    global l_avg, totalclasswise_pixel_acc, totalclasswise_gtpixels, totalclasswise_predpixels
    global steps

    model.train()
    
    if args.freeze:
        model.apply(set_bn_eval)

    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        assert images.size()[2:] == labels.size()[1:]
        # print('Inputs size =', images.size())
        # print('Labels size =', labels.size())

        if i % args.iter_size == 0:
            optimizer.zero_grad()
        
        outputs = model(images, labels)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size(1) == data.n_classes
        # print('Outputs size =', outputs.size())

        loss = criterion(outputs, labels)
        # print('loss:', loss)
        # sys.exit()
        
        total_valid_pixel = torch.sum(labels.data != criterion.ignore_index)
        classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([outputs], labels, data.n_classes)

        total_valid_pixel = torch.FloatTensor([total_valid_pixel]).to(device)
        classwise_pixel_acc = torch.FloatTensor([classwise_pixel_acc]).to(device)
        classwise_gtpixels = torch.FloatTensor([classwise_gtpixels]).to(device)
        classwise_predpixels = torch.FloatTensor([classwise_predpixels]).to(device)

        loss = loss / float(total_valid_pixel)
        loss = loss / float(args.iter_size)
        loss.backward()

        if i % args.iter_size == 0:
            optimizer.step()

        l_avg += loss.sum().data.cpu().numpy()
        steps += total_valid_pixel
        totalclasswise_pixel_acc += classwise_pixel_acc.sum(0).data.cpu().numpy()
        totalclasswise_gtpixels += classwise_gtpixels.sum(0).data.cpu().numpy()
        totalclasswise_predpixels += classwise_predpixels.sum(0).data.cpu().numpy()

        if (i + 1) % args.epoch_log_size == 0:
            print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1, args.epochs, loss.sum().item()))

        if (i + 1) % args.iter_size == 0:
            scheduler.step()

        if (i + 1) % args.log_size == 0:
            pickle.dump(images[0].cpu().numpy(),
                        open(os.path.join(ROOT_ADDRESS, "results/saved_train_images/" + str(epoch) + "_" + str(i) + "_input.p"), "wb"))

            pickle.dump(np.transpose(data.decode_segmap(outputs[0].data.cpu().numpy().argmax(0)), [2, 0, 1]),
                        open(os.path.join(ROOT_ADDRESS, "results/saved_train_images/" + str(epoch) + "_" + str(i) + "_output.p"), "wb"))

            pickle.dump(np.transpose(data.decode_segmap(labels[0].cpu().numpy()), [2, 0, 1]),
                        open(os.path.join(ROOT_ADDRESS, "results/saved_train_images/" + str(epoch) + "_" + str(i) + "_target.p"), "wb"))

def val(model, criterion, valloader, epoch, data):
    print('='*10, 'Validate step', '='*10, '\n')

    global l_avg_test, totalclasswise_pixel_acc_test, totalclasswise_gtpixels_test, totalclasswise_predpixels_test
    global steps_test

    model.eval()

    for i, (images, labels) in enumerate(valloader):
        images = images.to(device)
        labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images, labels)
        
        loss = criterion(outputs, labels)
        total_valid_pixel = torch.sum(labels.data != criterion.ignore_index)
        classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([outputs], labels, data.n_classes)

        total_valid_pixel = torch.FloatTensor([total_valid_pixel]).to(device)
        classwise_pixel_acc = torch.FloatTensor([classwise_pixel_acc]).to(device)
        classwise_gtpixels = torch.FloatTensor([classwise_gtpixels]).to(device)
        classwise_predpixels = torch.FloatTensor([classwise_predpixels]).to(device)

        l_avg_test += loss.sum().data.cpu().numpy()
        steps_test += total_valid_pixel
        totalclasswise_pixel_acc_test += classwise_pixel_acc.sum(0).data.cpu().numpy()
        totalclasswise_gtpixels_test += classwise_gtpixels.sum(0).data.cpu().numpy()
        totalclasswise_predpixels_test += classwise_predpixels.sum(0).data.cpu().numpy()

        if (i + 1) % 50 == 0:
            pickle.dump(images[0].cpu().numpy(),
                        open(os.path.join(ROOT_ADDRESS, "results/saved_val_images/" + str(epoch) + "_" + str(i) + "_input.p"), "wb"))

            pickle.dump(np.transpose(data.decode_segmap(outputs[0].data.cpu().numpy().argmax(0)), [2, 0, 1]),
                        open(os.path.join(ROOT_ADDRESS, "results/saved_val_images/" + str(epoch) + "_" + str(i) + "_output.p"), "wb"))

            pickle.dump(np.transpose(data.decode_segmap(labels[0].cpu().numpy()), [2, 0, 1]),
                        open(os.path.join(ROOT_ADDRESS, "results/saved_val_images/" + str(epoch) + "_" + str(i) + "_target.p"), "wb"))

    

if __name__ == '__main__':
    main(args)
