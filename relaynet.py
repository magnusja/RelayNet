import torch
import torch.nn as nn
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, num_input_channels=1, kernel=(3, 7), stride=1, num_output_channels=64):
        super().__init__()
        padding = (np.asarray(kernel) - 1) / 2
        padding = tuple(padding.astype(np.int))

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=num_input_channels, out_channels=num_output_channels,
                      kernel_size=kernel,
                      padding=padding,
                      stride=stride),
            nn.BatchNorm2d(num_features=num_output_channels),
            nn.PReLU()
        )

    def forward(self, input):
        return self.model(input)


class EncoderBlock(nn.Module):
    def __init__(self, num_input_channels=1, kernel=(3, 7), stride_conv=1, stride_pool=2, num_output_channels=64):
        super().__init__()
        self.basic = BasicBlock(num_input_channels, kernel, stride_conv, num_output_channels)
        self.pool = nn.MaxPool2d(kernel, stride_pool, return_indices=True)

    def forward(self, input):
        tmp = self.basic(input)
        out, indices = self.pool(tmp)
        return out, indices, tmp


class DecoderBlock(nn.Module):
    def __init__(self, num_input_channels=64, kernel=(3, 7), stride_conv=1, stride_pool=2, num_output_channels=64):
        super().__init__()
        self.basic = BasicBlock(num_input_channels * 2, kernel, stride_conv, num_output_channels)
        self.unpool = nn.MaxUnpool2d(kernel, stride_pool)

    def forward(self, input, indices, encoder_block):
        tmp = self.unpool(input, indices, output_size=encoder_block.size())
        tmp = torch.cat((encoder_block, tmp), dim=1)
        return self.basic(tmp)


class ClassifierBlock(nn.Module):
    def __init__(self, num_input_channels=64, kernel=(1, 1), stride_conv=1, num_classes=10):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Conv2d(num_input_channels, num_classes, kernel, stride_conv),
            nn.Softmax2d()
        )

    def forward(self, input):
        return self.classify(input)


class RelayNet(nn.Module):
    def __init__(self, num_input_channels=1, kernel=(3, 3), stride_conv=1, stride_pool=2, num_output_channels=64,
                 num_encoders=3, num_classes=10, kernel_classify=(1, 1)):
        super().__init__()
        self.encoders = nn.ModuleList([EncoderBlock(num_input_channels if i == 0 else num_output_channels, kernel,
                                                    stride_conv, stride_pool, num_output_channels)
                                       for i in range(num_encoders)])
        self.bottleneck = BasicBlock(num_output_channels, kernel, stride_conv, num_output_channels)
        self.decoders = nn.ModuleList(
            [DecoderBlock(num_output_channels, kernel, stride_conv, stride_pool, num_output_channels)
             for _ in range(num_encoders)])
        self.classify = ClassifierBlock(num_output_channels, kernel_classify, stride_conv, num_classes)

    def forward(self, input):
        out = input
        encodings = list()
        for encoder in self.encoders:
            out, indices, before_maxpool = encoder(out)
            encodings.append((out, indices, before_maxpool))

        out = self.bottleneck(encodings[-1][0])

        for i, encoded in enumerate(reversed(encodings)):
            decoder = self.decoders[i]
            out = decoder(out, encoded[1], encoded[2])

        return self.classify(out)
