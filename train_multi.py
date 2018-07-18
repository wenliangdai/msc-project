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
ROOT = '/home/wenlidai/sunets-reproduce/'
RESULT = 'results_'

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

    traindata = data_loader('train', n_classes=args.n_classes, transform=input_transform, target_transform=target_transform, do_transform=True)
    trainloader = data.DataLoader(traindata, batch_size=args.batch_size, num_workers=2, shuffle=True)
    valdata = data_loader('val', n_classes=args.n_classes, transform=input_transform, target_transform=target_transform)
    valloader = data.DataLoader(valdata, batch_size=args.batch_size, num_workers=2, shuffle=False)

    n_classes = traindata.n_classes
    n_trainsamples_total = len(traindata)
    n_trainsamples_lip = 28280
    n_trainsamples_sbd = n_trainsamples_total - n_trainsamples_lip
    n_iters_per_epoch_common = np.ceil(n_trainsamples_total / float(20))
    n_iters_per_epoch_lip = np.ceil(n_trainsamples_lip / float(10))
    n_iters_per_epoch_sbd = np.ceil(n_trainsamples_sbd / float(10))

    print('# total training samples = {}'.format(n_trainsamples_total))
    print('# sbd training samples = {}'.format(n_trainsamples_sbd))
    print('# lip training samples = {}'.format(n_trainsamples_lip))

    # Setup Model
    model = get_model(
        name=args.arch, 
        n_classes=n_classes, 
        # ignore_index=traindata.ignore_index, 
        output_stride=args.output_stride,
        pretrained=args.pretrained,
        momentum_bn=args.momentum_bn,
        dprob=args.dprob
    ).to(device)

    X=[]
    Y1=[]
    Y1_test=[]
    Y2=[]
    Y2_test=[]
    avg_pixel_acc = 0
    mean_class_acc = 0
    mIoU = 0
    avg_pixel_acc_test = 0
    mean_class_acc_test = 0
    mIoU_test = 0

    # Learning rates: For new layers (such as final layer), we set lr to be 10x the learning rate of layers already trained
    common_bias = filter(lambda x: ('bias' in x[0]) and ('final' not in x[0]), model.named_parameters())
    common_bias = list(map(lambda x: x[1], common_bias))
    common_nonbias = filter(lambda x: ('bias' not in x[0]) and ('final' not in x[0]), model.named_parameters())
    common_nonbias = list(map(lambda x: x[1], common_nonbias))

    final1_bias = filter(lambda x: ('bias' in x[0]) and ('final1' in x[0]) and ('conv' in x[0]), model.named_parameters())
    final1_bias = list(map(lambda x: x[1], final1_bias))
    final1_nonbias = filter(lambda x: (('bias' not in x[0]) or ('bn' in x[0])) and ('final1' in x[0]), model.named_parameters())
    final1_nonbias = list(map(lambda x: x[1], final1_nonbias))
    
    final2_bias = filter(lambda x: ('bias' in x[0]) and ('final2' in x[0]) and ('conv' in x[0]), model.named_parameters())
    final2_bias = list(map(lambda x: x[1], final2_bias))
    final2_nonbias = filter(lambda x: (('bias' not in x[0]) or ('bn' in x[0])) and ('final2' in x[0]), model.named_parameters())
    final2_nonbias = list(map(lambda x: x[1], final2_nonbias))

    optimizer_common = torch.optim.SGD(
        [{'params': common_bias, 'lr': args.lr},
         {'params': common_nonbias, 'lr': args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        nesterov=(args.optim == 'Nesterov'))
    optimizer_sbd = torch.optim.SGD(
        [{'params': final1_bias, 'lr': args.lr},
         {'params': final1_nonbias, 'lr': args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        nesterov=(args.optim == 'Nesterov'))
    optimizer_lip = torch.optim.SGD(
        [{'params': final2_bias, 'lr': args.lr},
         {'params': final2_nonbias, 'lr': args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        nesterov=(args.optim == 'Nesterov'))

    optimizers = [optimizer_common, optimizer_sbd, optimizer_lip]

    # Setting up scheduler
    total_iters_common = (args.epochs * n_iters_per_epoch_common)
    total_iters_sbd = (args.epochs * n_iters_per_epoch_sbd)
    total_iters_lip = (args.epochs * n_iters_per_epoch_lip)
    lambda_common = lambda step: 0.5 + 0.5 * math.cos(np.pi * step / total_iters_common)
    lambda_sbd = lambda step: 0.5 + 0.5 * math.cos(np.pi * step / total_iters_sbd)
    lambda_lip = lambda step: 0.5 + 0.5 * math.cos(np.pi * step / total_iters_lip)
    scheduler_common = lr_scheduler.LambdaLR(optimizer_common, lr_lambda=[lambda_common]*2)
    scheduler_sbd = lr_scheduler.LambdaLR(optimizer_sbd, lr_lambda=[lambda_sbd]*2)
    scheduler_lip = lr_scheduler.LambdaLR(optimizer_lip, lr_lambda=[lambda_lip]*2)

    schedulers = [scheduler_common, scheduler_sbd, scheduler_lip]

    global l_avg, totalclasswise_pixel_acc, totalclasswise_gtpixels, totalclasswise_predpixels
    global l_avg_test, totalclasswise_pixel_acc_test, totalclasswise_gtpixels_test, totalclasswise_predpixels_test
    global steps, steps_test

    scheduler_common.step()
    scheduler_sbd.step()
    scheduler_lip.step()

    counter_sizes = [20, 10, 10]
    global counters
    counters = [0, 0, 0]

    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=traindata.ignore_index)

    for epoch in range(args.epochs):
        print('='*10, 'Epoch %d' % (epoch + 1), '='*10)
        l_avg = [0, 0]
        totalclasswise_pixel_acc = [0, 0]
        totalclasswise_gtpixels = [0, 0]
        totalclasswise_predpixels = [0, 0]
        l_avg_test = [0, 0]
        totalclasswise_pixel_acc_test = [0, 0]
        totalclasswise_gtpixels_test = [0, 0]
        totalclasswise_predpixels_test = [0, 0]
        steps = [0, 0]
        steps_test = [0, 0]
        
        train(model, optimizers, criterion, trainloader, epoch, schedulers, traindata, counter_sizes)
        val(model, criterion, valloader, epoch, valdata)

        # save the model every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            if (epoch + 1) > 5:
                os.remove(os.path.join(ROOT, RESULT, "{}_{}_{}.pkl".format(args.arch, args.dataset, epoch - 4)))
                os.remove(os.path.join(ROOT, RESULT, "{}_{}_{}_optimizer0.pkl".format(args.arch, args.dataset, epoch - 4)))
                os.remove(os.path.join(ROOT, RESULT, "{}_{}_{}_optimizer1.pkl".format(args.arch, args.dataset, epoch - 4)))
                os.remove(os.path.join(ROOT, RESULT, "{}_{}_{}_optimizer2.pkl".format(args.arch, args.dataset, epoch - 4)))
            torch.save(model, os.path.join(ROOT, RESULT, "{}_{}_{}.pkl".format(args.arch, args.dataset, epoch + 1)))
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizers[0].state_dict()},
                       os.path.join(ROOT, RESULT, "{}_{}_{}_optimizer0.pkl".format(args.arch, args.dataset, epoch + 1)))
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizers[1].state_dict()},
                       os.path.join(ROOT, RESULT, "{}_{}_{}_optimizer1.pkl".format(args.arch, args.dataset, epoch + 1)))
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizers[2].state_dict()},
                       os.path.join(ROOT, RESULT, "{}_{}_{}_optimizer2.pkl".format(args.arch, args.dataset, epoch + 1)))
        
        # remove old loss & accuracy files
        if os.path.isfile(os.path.join(ROOT, RESULT, "saved_loss.p")):
            os.remove(os.path.join(ROOT, RESULT, "saved_loss.p"))
        if os.path.isfile(os.path.join(ROOT, RESULT, "saved_accuracy.p")):
            os.remove(os.path.join(ROOT, RESULT, "saved_accuracy.p"))

        # save train and validation loss
        X.append(epoch + 1)
        Y1.append(l_avg[0] / steps[0])
        Y1_test.append(l_avg_test[0] / steps_test[0])
        Y2.append(l_avg[1] / steps[1])
        Y2_test.append(l_avg_test[1] / steps_test[1])
        saved_loss={"X": X, "Y1": Y1, "Y2": Y2, "Y1_test": Y1_test, "Y2_test": Y2_test}
        pickle.dump(saved_loss, open(os.path.join(ROOT, RESULT, "saved_loss.p"), "wb"))
        
        # pixel accuracy
        totalclasswise_pixel_acc[0] = totalclasswise_pixel_acc[0].reshape((-1, n_classes[0])).astype(np.float32)
        totalclasswise_gtpixels[0] = totalclasswise_gtpixels[0].reshape((-1, n_classes[0]))
        totalclasswise_predpixels[0] = totalclasswise_predpixels[0].reshape((-1, n_classes[0]))
        totalclasswise_pixel_acc_test[0] = totalclasswise_pixel_acc_test[0].reshape((-1, n_classes[0])).astype(np.float32)
        totalclasswise_gtpixels_test[0] = totalclasswise_gtpixels_test[0].reshape((-1, n_classes[0]))
        totalclasswise_predpixels_test[0] = totalclasswise_predpixels_test[0].reshape((-1, n_classes[0]))

        totalclasswise_pixel_acc[1] = totalclasswise_pixel_acc[1].reshape((-1, n_classes[1])).astype(np.float32)
        totalclasswise_gtpixels[1] = totalclasswise_gtpixels[1].reshape((-1, n_classes[1]))
        totalclasswise_predpixels[1] = totalclasswise_predpixels[1].reshape((-1, n_classes[1]))
        totalclasswise_pixel_acc_test[1] = totalclasswise_pixel_acc_test[1].reshape((-1, n_classes[1])).astype(np.float32)
        totalclasswise_gtpixels_test[1] = totalclasswise_gtpixels_test[1].reshape((-1, n_classes[1]))
        totalclasswise_predpixels_test[1] = totalclasswise_predpixels_test[1].reshape((-1, n_classes[1]))

        if isinstance(avg_pixel_acc, np.ndarray):
            avg_pixel_acc[0] = np.vstack((avg_pixel_acc[0], np.sum(totalclasswise_pixel_acc[0], axis=1) / np.sum(totalclasswise_gtpixels[0], axis=1)))
            mean_class_acc[0] = np.vstack((mean_class_acc[0], np.mean(totalclasswise_pixel_acc[0] / totalclasswise_gtpixels[0], axis=1)))
            mIoU[0] = np.vstack((mIoU[0], np.mean(totalclasswise_pixel_acc[0] / (totalclasswise_gtpixels[0] + totalclasswise_predpixels[0] - totalclasswise_pixel_acc[0]), axis=1)))
            avg_pixel_acc[1] = np.vstack((avg_pixel_acc[1], np.sum(totalclasswise_pixel_acc[1], axis=1) / np.sum(totalclasswise_gtpixels[1], axis=1)))
            mean_class_acc[1] = np.vstack((mean_class_acc[1], np.mean(totalclasswise_pixel_acc[1] / totalclasswise_gtpixels[1], axis=1)))
            mIoU[1] = np.vstack((mIoU[1], np.mean(totalclasswise_pixel_acc[1] / (totalclasswise_gtpixels[1] + totalclasswise_predpixels[1] - totalclasswise_pixel_acc[1]), axis=1)))

            avg_pixel_acc_test[0] = np.vstack((avg_pixel_acc_test[0], np.sum(totalclasswise_pixel_acc_test[0],axis=1) / np.sum(totalclasswise_gtpixels_test[0], axis=1)))
            mean_class_acc_test[0] = np.vstack((mean_class_acc_test[0], np.mean(totalclasswise_pixel_acc_test[0] / totalclasswise_gtpixels_test[0], axis=1)))
            mIoU_test[0] = np.vstack((mIoU_test[0], np.mean(totalclasswise_pixel_acc_test[0] / (totalclasswise_gtpixels_test[0] + totalclasswise_predpixels_test[0] - totalclasswise_pixel_acc_test[0]), axis=1)))
            avg_pixel_acc_test[1] = np.vstack((avg_pixel_acc_test[1], np.sum(totalclasswise_pixel_acc_test[1],axis=1) / np.sum(totalclasswise_gtpixels_test[1], axis=1)))
            mean_class_acc_test[1] = np.vstack((mean_class_acc_test[1], np.mean(totalclasswise_pixel_acc_test[1] / totalclasswise_gtpixels_test[1], axis=1)))
            mIoU_test[1] = np.vstack((mIoU_test[1], np.mean(totalclasswise_pixel_acc_test[1] / (totalclasswise_gtpixels_test[1] + totalclasswise_predpixels_test[1] - totalclasswise_pixel_acc_test[0]), axis=1)))
        else:
            avg_pixel_acc = []
            mean_class_acc = []
            mIoU = []
            avg_pixel_acc[0] = np.sum(totalclasswise_pixel_acc[0], axis=1) / np.sum(totalclasswise_gtpixels[0], axis=1)
            mean_class_acc[0] = np.mean(totalclasswise_pixel_acc[0] / totalclasswise_gtpixels[0], axis=1)
            mIoU[0] = np.mean(totalclasswise_pixel_acc[0] / (totalclasswise_gtpixels[0] + totalclasswise_predpixels[0] - totalclasswise_pixel_acc[0]), axis=1)
            avg_pixel_acc[1] = np.sum(totalclasswise_pixel_acc[1], axis=1) / np.sum(totalclasswise_gtpixels[1], axis=1)
            mean_class_acc[1] = np.mean(totalclasswise_pixel_acc[1] / totalclasswise_gtpixels[1], axis=1)
            mIoU[1] = np.mean(totalclasswise_pixel_acc[1] / (totalclasswise_gtpixels[1] + totalclasswise_predpixels[1] - totalclasswise_pixel_acc[1]), axis=1)

            avg_pixel_acc_test = []
            mean_class_acc_test = []
            mIoU_test = []
            avg_pixel_acc_test[0] = np.sum(totalclasswise_pixel_acc_test[0], axis=1) / np.sum(totalclasswise_gtpixels_test[0], axis=1)
            mean_class_acc_test[0] = np.mean(totalclasswise_pixel_acc_test[0] / totalclasswise_gtpixels_test[0], axis=1)
            mIoU_test[0] = np.mean(totalclasswise_pixel_acc_test[0] / (totalclasswise_gtpixels_test[0] + totalclasswise_predpixels_test[0] - totalclasswise_pixel_acc_test[0]), axis=1)
            avg_pixel_acc_test[1] = np.sum(totalclasswise_pixel_acc_test[1], axis=1) / np.sum(totalclasswise_gtpixels_test[1], axis=1)
            mean_class_acc_test[1] = np.mean(totalclasswise_pixel_acc_test[1] / totalclasswise_gtpixels_test[1], axis=1)
            mIoU_test[1] = np.mean(totalclasswise_pixel_acc_test[1] / (totalclasswise_gtpixels_test[1] + totalclasswise_predpixels_test[1] - totalclasswise_pixel_acc_test[1]), axis=1)

        saved_accuracy = {
            "X": X, 
            "P1": avg_pixel_acc[0], "P2": avg_pixel_acc[1], 
            "M1": mean_class_acc[0], "M2": mean_class_acc[1], 
            "I1": mIoU[0], "I2": mIoU[1],
            "P1_test": avg_pixel_acc_test[0], "P2_test": avg_pixel_acc_test[1],
            "M1_test": mean_class_acc_test[0], "M2_test": mean_class_acc_test[1], 
            "I1_test": mIoU_test[0], "I2_test": mIoU_test[1]
        }
        pickle.dump(saved_accuracy, open(os.path.join(ROOT, RESULT, "saved_accuracy.p"), "wb"))

        # save the best model
        this_mIoU1 = np.mean(totalclasswise_pixel_acc_test[0] / (totalclasswise_gtpixels_test[0] + totalclasswise_predpixels_test[0] - totalclasswise_pixel_acc_test[0]), axis=1)[0]
        this_mIoU2 = np.mean(totalclasswise_pixel_acc_test[1] / (totalclasswise_gtpixels_test[1] + totalclasswise_predpixels_test[1] - totalclasswise_pixel_acc_test[1]), axis=1)[0]
        print('Val: mIoU_sbd = {}, mIoU_lip = {}'.format(this_mIoU1, this_mIoU2))

# Incase one want to freeze BN params
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def train(model, optimizers, criterion, trainloader, epoch, schedulers, data, counter_sizes):
    global l_avg, totalclasswise_pixel_acc, totalclasswise_gtpixels, totalclasswise_predpixels
    global steps
    global counters

    model.train()
    
    if args.freeze:
        model.apply(set_bn_eval)

    for i, (images, labels, task) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        # Increment common CNN's counter for each image
        counters[0] += 1
        if task == 0:
            # if is sbd image
            counters[1] += 1
        else:
            # if is lip image
            counters[2] += 1
            
        outputs = model(images, task)
        loss = criterion(outputs, labels)
        
        total_valid_pixel = torch.sum(labels.data != criterion.ignore_index)
        classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([outputs], labels, data.n_classes[task])

        total_valid_pixel = torch.FloatTensor([total_valid_pixel]).to(device)
        classwise_pixel_acc = torch.FloatTensor([classwise_pixel_acc]).to(device)
        classwise_gtpixels = torch.FloatTensor([classwise_gtpixels]).to(device)
        classwise_predpixels = torch.FloatTensor([classwise_predpixels]).to(device)

        total_valid_pixel = float(total_valid_pixel.sum(0).data.cpu().numpy())

        total_loss = loss.sum()
        total_loss = total_loss / float(total_valid_pixel)
        total_loss = total_loss / float(args.iter_size)
        total_loss.backward()

        for i in range(3):
            if counters[i] % counter_sizes[i] == 0:
                optimizers[i].step()
                optimizers[i].zero_grad()
                schedulers[i].step()

        l_avg[task] += loss.sum().data.cpu().numpy()
        steps[task] += total_valid_pixel
        totalclasswise_pixel_acc[task] += classwise_pixel_acc.sum(0).data.cpu().numpy()
        totalclasswise_gtpixels[task] += classwise_gtpixels.sum(0).data.cpu().numpy()
        totalclasswise_predpixels[task] += classwise_predpixels.sum(0).data.cpu().numpy()

        if (i + 1) % args.log_size == 0:
            pickle.dump(images[0].cpu().numpy(),
                        open(os.path.join(ROOT, RESULT, "saved_train_images/" + str(epoch) + "_" + str(i) + "_input.p"), "wb"))

            pickle.dump(np.transpose(data.decode_segmap(outputs[0].data.cpu().numpy().argmax(0)), [2, 0, 1]),
                        open(os.path.join(ROOT, RESULT, "saved_train_images/" + str(epoch) + "_" + str(i) + "_output.p"), "wb"))

            pickle.dump(np.transpose(data.decode_segmap(labels[0].cpu().numpy()), [2, 0, 1]),
                        open(os.path.join(ROOT, RESULT, "saved_train_images/" + str(epoch) + "_" + str(i) + "_target.p"), "wb"))

def val(model, criterion, valloader, epoch, data):
    global l_avg_test, totalclasswise_pixel_acc_test, totalclasswise_gtpixels_test, totalclasswise_predpixels_test
    global steps_test

    model.eval()

    for i, (images, labels, task) in enumerate(valloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images, task)
            loss = criterion(outputs, labels)

            total_valid_pixel = torch.sum(labels.data != criterion.ignore_index)
            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([outputs], labels, data.n_classes[task])

            total_valid_pixel = torch.FloatTensor([total_valid_pixel]).to(device)
            classwise_pixel_acc = torch.FloatTensor([classwise_pixel_acc]).to(device)
            classwise_gtpixels = torch.FloatTensor([classwise_gtpixels]).to(device)
            classwise_predpixels = torch.FloatTensor([classwise_predpixels]).to(device)

            total_valid_pixel = float(total_valid_pixel.sum(0).data.cpu().numpy())

            l_avg_test[task] += loss.sum().data.cpu().numpy()
            steps_test[task] += total_valid_pixel
            totalclasswise_pixel_acc_test[task] += classwise_pixel_acc.sum(0).data.cpu().numpy()
            totalclasswise_gtpixels_test[task] += classwise_gtpixels.sum(0).data.cpu().numpy()
            totalclasswise_predpixels_test[task] += classwise_predpixels.sum(0).data.cpu().numpy()

            if (i + 1) % 100 == 0:
                pickle.dump(images[0].cpu().numpy(),
                            open(os.path.join(ROOT, RESULT, "saved_val_images/" + str(epoch) + "_" + str(i) + "_input.p"), "wb"))

                pickle.dump(np.transpose(data.decode_segmap(outputs[0].data.cpu().numpy().argmax(0)), [2, 0, 1]),
                            open(os.path.join(ROOT, RESULT, "saved_val_images/" + str(epoch) + "_" + str(i) + "_output.p"), "wb"))

                pickle.dump(np.transpose(data.decode_segmap(labels[0].cpu().numpy()), [2, 0, 1]),
                            open(os.path.join(ROOT, RESULT, "saved_val_images/" + str(epoch) + "_" + str(i) + "_target.p"), "wb"))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='sunet64_multi',
                        help='Architecture to use [\'sunet64, sunet128, sunet7128 etc\']')
    parser.add_argument('--model_path', help='Path to the saved model', type=str)
    parser.add_argument('--best_model_path', help='Path to the saved best model', type=str)
    parser.add_argument('--dataset', nargs='?', type=str, default='sbd',
                        help='Dataset to use [\'sbd, coco, cityscapes etc\']')
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
    parser.add_argument('--dprob', nargs='?', type=float, default=1e-7,
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
    parser.add_argument('--n_classes', nargs='?', type=list, default=[21,20],
                        help='number of classes of the labels')
    parser.add_argument('--optim', nargs='?', type=str, default='SGD',
                        help='Optimizer to use [\'SGD, Nesterov etc\']')

    global args
    args = parser.parse_args()

    args.n_classes = [int(n) for n in args.n_classes]

    RESULT = RESULT + args.dataset
    if args.pretrained:
        RESULT = RESULT + '_pretrained'
    
    main(args)
