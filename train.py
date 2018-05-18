import argparse

import torch
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange


from data_utils import get_imdb_data
from loss import TotalLoss, dice_coeff
from relaynet import RelayNet


def train(epoch, data, net, criterion, optimizer, args):
    train_set = DataLoader(data, batch_size=args.batch_size, num_workers=0, shuffle=True)

    progress_bar = tqdm(iter(train_set))
    moving_loss = 0

    net.train()
    for img, label, label_bin, weight in progress_bar:
        img, label, label_bin, weight = Variable(img), Variable(label), Variable(label_bin), Variable(weight)
        label = label.type(torch.LongTensor)
        label_bin = label_bin.type(torch.FloatTensor)

        if args.cuda:
            img, label, label_bin, weight = img.cuda(), label.cuda(), label_bin.cuda(), weight.cuda()

        output = net(img)
        loss = criterion(output, label, weight, label_bin)
        net.zero_grad()
        loss.backward()
        optimizer.step()

        if moving_loss == 0:
            moving_loss = loss.item()
        else:
            moving_loss = moving_loss * 0.9 + loss.item() * 0.1

        dice_avg = torch.mean(dice_coeff(output, label_bin))

        progress_bar.set_description(
            'Epoch: {}; Loss: {:.5f}; Avg: {:.5f}; Dice: {:.5f}'
                .format(epoch + 1, loss.item(), moving_loss, dice_avg.item()))


def valid(data, net, args):
    valid_set = DataLoader(data, batch_size=args.batch_size, num_workers=0, shuffle=True)
    net.eval()

    progress_bar = tqdm(iter(valid_set))

    dice_avg = list()
    for img, label, label_bin, weight in progress_bar:
        img, label, label_bin, weight = Variable(img), Variable(label), Variable(label_bin), Variable(weight)
        label_bin = label_bin.type(torch.FloatTensor)

        if args.cuda:
            img, label_bin= img.cuda(), label_bin.cuda()

        output = net(img)
        dice_avg.append(torch.mean(dice_coeff(output, label_bin)).item())

    dice_avg = np.asarray(dice_avg).mean()

    print('Validation dice avg: {}'.format(dice_avg))


def parse_args():
    parser = argparse.ArgumentParser(description='Train SqueezeNet with PyTorch.')
    parser.add_argument('--batch-size', action='store', type=int, dest='batch_size', default=8)
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=90)
    parser.add_argument('--cuda', action='store', type=bool, dest='cuda', default=True)
    parser.add_argument('--validation', action='store', type=bool, dest='validation', default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    relay_net = RelayNet()
    if args.cuda:
        relay_net = relay_net.cuda()

    criterion = TotalLoss()
    optimizer = optim.Adam(relay_net.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

    train_data, valid_data = get_imdb_data()

    for epoch in trange(args.epochs):
        scheduler.step(epoch)
        train(epoch, train_data, relay_net, criterion, optimizer, args)
        if args.validation:
            valid(valid_data, relay_net, args)


if __name__ == '__main__':
    main()
