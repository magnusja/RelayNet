import multiprocessing
from itertools import product

import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import dice_coeff

def iter_data_and_predict(data, net, args, mc_samples=10):
    valid_set = DataLoader(data, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True)
    net.eval()

    progress_bar = tqdm(iter(valid_set))

    for img, label, label_bin, weight in progress_bar:
        img, label, label_bin, weight = Variable(img), Variable(label), Variable(label_bin), Variable(weight)
        label = label.type(torch.LongTensor)

        if args.cuda:
            img, label = img.cuda(), label.cuda()

        avg, per_class_entropy, overall_entropy, samples = net.predict(img, times=mc_samples)

        yield avg, per_class_entropy, overall_entropy, samples, label


def error_map_dice(data, net, args, mc_samples=10, entropy_threshold=0.5):
    """
    Computes the dice score between prediction error map and the entropy. This is a measure on how well the entropy
    describes the actual error the network makes.
    :param data:
    :param net:
    :param args:
    :param mc_samples:
    :param entropy_threshold:
    :return:
    """

    dice_avg = list()
    for avg, _, overall_entropy, _, label in iter_data_and_predict(data, net, args, mc_samples):

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


def structure_wise_uncertainty_dice(data, net, args, mc_samples=10, n_classes=9):

    dice_avg = list()

    for _, _, _, samples, _ in iter_data_and_predict(data, net, args, mc_samples):

        samples = torch.Tensor(samples)
        if args.cuda:
            samples = samples.cuda()

        for i, j in product(range(mc_samples), range(mc_samples)):
            if i == j:
                continue

            dice_score = dice_coeff(samples[i], samples[j, :], n_classes=n_classes)
            dice_avg.append(torch.mean(dice_score, dim=0).cpu().numpy())

    dice_avg = np.asarray(dice_avg).mean(axis=0)

    return dice_avg


def structure_wise_uncertainty_cv(data, net, args, mc_samples=10):
    """
    Coefficient of variance = mean/std_dev
    :param data:
    :param net:
    :param args:
    :param mc_samples:
    :return:
    """
    cvs = list()

    for avg, _, _, samples, _ in iter_data_and_predict(data, net, args, mc_samples):

        std_dev = samples.std(axis=0)
        cv = avg / (std_dev + 1e-6)

        for x in cv:
            cvs.append(x)

    cv = np.asarray(cvs).mean(axis=(0, 2, 3))

    return cv