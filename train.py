import argparse
import os
import multiprocessing

import torch
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange


from data_utils import get_imdb_data
from loss import TotalLoss, dice_coeff
from relaynet import RelayNet, DenseBlock, BasicBlock


def train(epoch, data, net, criterion, optimizer, args):
    train_set = DataLoader(data, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True)

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


def valid(data, net, args, mc_samples=1):
    valid_set = DataLoader(data, batch_size=args.batch_size // 2, num_workers=multiprocessing.cpu_count(), shuffle=True)
    net.eval()

    progress_bar = tqdm(iter(valid_set))

    dice_avg = list()
    for img, label, label_bin, weight in progress_bar:
        img, label, label_bin, weight = Variable(img), Variable(label), Variable(label_bin), Variable(weight)
        label_bin = label_bin.type(torch.FloatTensor)

        if args.cuda:
            img, label_bin = img.cuda(), label_bin.cuda()

        if mc_samples > 1:
            # lol this is insanely inefficient
            avg, _, _ = net.predict(img, times=mc_samples)
            output = Variable(torch.Tensor(avg))
            if args.cuda:
                output = output.cuda()
        else:
            output = net(img)

        dice_avg.append(torch.mean(dice_coeff(output, label_bin)).item())

    dice_avg = np.asarray(dice_avg).mean()

    print('Validation dice avg: {}'.format(dice_avg))

    return dice_avg


def parse_args():
    parser = argparse.ArgumentParser(description='Train SqueezeNet with PyTorch.')
    parser.add_argument('--batch-size', action='store', type=int, dest='batch_size', default=8)
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=90)
    parser.add_argument('--cuda', action='store', type=bool, dest='cuda', default=True)
    parser.add_argument('--validation', action='store_true', dest='validation', default=True)
    parser.add_argument('--model-checkpoint-dir', action='store', type=str, default='./models')
    parser.add_argument('--use-dense-connections', action='store_true', dest='dense', default=False)
    parser.add_argument('--dropout-prob', action='store', type=float, default=0.5)

    return parser.parse_args()


def main():
    print('number of cpus used for loading data: {}'.format(multiprocessing.cpu_count()))
    args = parse_args()
    os.makedirs(args.model_checkpoint_dir, exist_ok=True)

    relay_net = RelayNet(basic_block=DenseBlock if args.dense else BasicBlock, dropout_prob=args.dropout_prob)
    print(relay_net)
    if args.cuda:
        relay_net = relay_net.cuda()

    criterion = TotalLoss()
    optimizer = optim.Adam(relay_net.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

    train_data, valid_data = get_imdb_data()

    for epoch in trange(args.epochs):
        scheduler.step(epoch)
        train(epoch, train_data, relay_net, criterion, optimizer, args)

        torch.save(relay_net.state_dict(), os.path.join(args.model_checkpoint_dir, 'model-{}.model'.format(epoch)))

    del criterion, optimizer, scheduler

    if args.validation:
        best = (-1, -1)
        for epoch in trange(args.epochs):
            relay_net.load_state_dict(torch.load(os.path.join(args.model_checkpoint_dir, 'model-{}.model'.format(epoch))))
            if args.cuda:
                relay_net = relay_net.cuda()
            dice = valid(valid_data, relay_net, args)
            _, best_dice = best

            if dice > best_dice:
                best = (epoch, dice)

        print('Best model with epoch {} and dice {}'.format(*best))


if __name__ == '__main__':
    main()
