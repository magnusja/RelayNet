import multiprocessing

import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import dice_coeff


def error_map_dice(data, net, args, mc_samples=10, entropy_threshold=0.5):
    """
    Comput the dice score between prediction error map and the entropy. This is a measure on how well the entropy
    describes the actual error the network makes.
    :param data:
    :param net:
    :param args:
    :param mc_samples:
    :param entropy_threshold:
    :return:
    """
    valid_set = DataLoader(data, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True)
    net.eval()

    progress_bar = tqdm(iter(valid_set))

    dice_avg = list()
    for img, label, label_bin, weight in progress_bar:
        img, label, label_bin, weight = Variable(img), Variable(label), Variable(label_bin), Variable(weight)
        label = label.type(torch.LongTensor)

        if args.cuda:
            img, label = img.cuda(), label.cuda()

        avg, _, overall_entropy = net.predict(img, times=mc_samples)

        overall_entropy = overall_entropy > entropy_threshold
        overall_entropy = Variable(torch.Tensor(overall_entropy.astype(np.float32)))

        indices = np.argmax(avg, axis=1)  # 1 is class dim
        indices = Variable(torch.LongTensor(indices))

        if args.cuda:
            overall_entropy, indices = overall_entropy.cuda(), indices.cuda()

        error_map = label != indices
        error_map = error_map.type(torch.cuda.FloatTensor if args.cuda else torch.FloatTensor)

        dice_avg.append(torch.mean(dice_coeff(overall_entropy, error_map, n_classes=1)).item())

    dice_avg = np.asarray(dice_avg).mean()

    print('dice avg: {}'.format(dice_avg))

    return dice_avg