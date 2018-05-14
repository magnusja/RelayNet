import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def forward(self, output, target_bin):
        batch_size = output.size()[0]
        upper = (output * target_bin).view(batch_size, 10, -1).sum(dim=-1)
        lower = (output ** 2).view(batch_size, 10, -1).sum(dim=-1) \
            + (target_bin ** 2).view(batch_size, 10, -1).sum(dim=-1)
        loss = 2 * upper / lower
        return torch.mean(1 - loss)


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

