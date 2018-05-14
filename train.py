import argparse

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange


from data_utils import get_imdb_data
from loss import WeightedClassificationLoss, TotalLoss
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
            moving_loss = loss.data[0]
        else:
            moving_loss = moving_loss * 0.9 + loss.data[0] * 0.1

        progress_bar.set_description(
            'Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(epoch + 1, loss.data[0], moving_loss))


def valid(data, squeeze_net, args):
    valid_set = DataLoader(data, batch_size=args.batch_size, num_workers=4, shuffle=True)
    squeeze_net.eval()

    acc = 0
    for x, y in tqdm(valid_set):
        x, y = Variable(x), Variable(y)

        if args.cuda:
            x, y = Variable(x).cuda(), Variable(y).cuda()

        output = squeeze_net(x)
        acc += y.eq(output > 0.85).sum() / y.size()[0]

    print('Validation accuracy: {}'.format(acc))


def parse_args():
    parser = argparse.ArgumentParser(description='Train SqueezeNet with PyTorch.')
    parser.add_argument('--batch-size', action='store', type=int, dest='batch_size', default=8)
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=90)
    parser.add_argument('--cuda', action='store', type=bool, dest='cuda', default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    relay_net = RelayNet()
    if args.cuda:
        relay_net = relay_net.cuda()

    criterion = TotalLoss()
    optimizer = optim.Adam(relay_net.parameters(), lr=0.003)

    train_data, valid_data = get_imdb_data()

    for epoch in trange(args.epochs):
        train(epoch, train_data, relay_net, criterion, optimizer, args)
        #valid(valid_data, relay_net, args)


if __name__ == '__main__':
    main()
