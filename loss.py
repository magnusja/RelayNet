import torch.nn as nn
import torch
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def forward(self, output, target_bin):
        if output.is_cuda:
            loss = Variable(torch.FloatTensor(1).cuda().zero_())
        else:
            loss = Variable(torch.FloatTensor(1).zero_())

        loss += 2 * torch.sum(output * target_bin) / (torch.sum(output ** 2) + torch.sum(target_bin ** 2))

        return (1 - loss) / (output.size()[0] + 1)


class WeightedClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.NLLLoss(size_average=True)

    def forward(self, output, target, weight):
        loss = self.criterion(output, target)

        return torch.mean(loss * weight)


class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.classification_loss = WeightedClassificationLoss()

    def forward(self, output, target, weight, target_bin):
        return self.classification_loss(output, target, weight) + self.dice_loss(output, target_bin)

