#------------------------------------------
# written by Leonard Niu
# HIT
#------------------------------------------

import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable

from model import VGG16

parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--model', type=str, default='VGG16', help='Net')
parser.add_argument('--bs', default=8, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', default=False, help='resume from checkpoint')
parser.add_argument('--GPU', default=True, help='train with GPU')
parser.add_argument('--epoch', default=10, type=int, help='epoch')
parser.add_argument('--input_size', default=224, type=int)
parser.add_argument('--ckpt_path', default='./checkpoints/VGG16-165.t7')
parser.add_argument('--data_path', default='./data')

cfg, unknown = parser.parse_known_args()

start_epoch = 0
best_test_acc = 0
best_test_acc_epoch = 0

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(cfg.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)

transform_test = transforms.Compose(
    [
        transforms.TenCrop(cfg.input_size),
        transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ]
)

train_data = torchvision.datasets.ImageFolder(
    os.path.join(cfg.data_path, 'train'), transform=transform_train
)

test_data = torchvision.datasets.ImageFolder(
    os.path.join(cfg.data_path, 'test'), transform=transform_test
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=cfg.bs, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=cfg.bs, shuffle=True
)

net = VGG16.Net()

if cfg.resume:
    print('------------------------------')
    print('==> Loading the checkpoint ')
    if not os.path.exists(cfg.ckpt_path):
        raise AssertionError['Can not find path']
    checkpoint = torch.load(cfg.ckpt_path)
    net.load_state_dict(checkpoint['net'])
    best_test_acc = checkpoint['best_test_acc']
    start_epoch = checkpoint['best_test_acc_epoch'] + 1
else:
    print('------------------------------')
    print('==> Building new model ')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)


def train(Epoch):
    global train_acc
    # net.train()
    trainloss = 0.0
    total = 0
    correct = 0
    for i, (inputs, target) in enumerate(train_loader):
        # print(i, inputs.size(), target.size())
        inputs, target = Variable(inputs), Variable(target)
        if use_cuda:
            inputs = inputs.to(device)
            target = target.to(device)
        outputs = net(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, target)
        # loss.requires_grad=True
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)

        correct += predicted.eq(target.data).sum()

        print("In %d epoch, the %s forward and backward pass done." % (epoch,i))

        # trainacc = 1.0 * correct/total

    train_acc = 100.0 * int(correct.data) / total
    print('%d training epoch is over'%epoch)
    print('In %d pictures, %d are predicted right' %(total, correct))
    print('Training acc is %.4f%%'%train_acc)
    # print ("Training process is over")


def test(Epoch):
    global test_acc
    global best_test_acc
    global best_test_acc_epoch
    net.eval()
    total = 0
    correct = 0
    for i, (inputs, target) in enumerate(test_loader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, target = Variable(inputs), Variable(target)
        if use_cuda:
            inputs = inputs.to(device)
            target = target.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

        _, predicted = torch.max(outputs_avg.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).sum()

    test_acc = 100.0 * correct / total
    print('One test epoch is over')
    print('Testing acc is %.3f' % test_acc)
    if test_acc > best_test_acc:
        print('Saving new ckpt')
        print("best_test_acc: %0.3f" % test_acc)
        print('best_test_epoch: %d ' % Epoch)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'besttestacc': test_acc,
            'besttestaccepoch': Epoch,
        }
        torch.save(state, os.path.join('./checkpoints', cfg.model + '-' + str(Epoch) + '.t7'))
        best_test_acc = test_acc
        best_test_acc_epoch = Epoch
        print('Update over')
    # print ("Test process is over")


if __name__ == '__main__':

    print('--------------------------------------------------------')
    print('-------------batch size: %d' % cfg.bs)
    print('-------------backbone: %s' % cfg.model)
    print('-------------total epoch: %d' % cfg.epoch)
    print('-------------device: %s' % ('GPU'if cfg.GPU else 'CPU'))
    print('-------------input size: %d x %d' % (cfg.input_size, cfg.input_size))
    print('--------------------------------------------------------')
    for epoch in range(start_epoch, cfg.epoch):
        train(epoch)
        test(epoch)
