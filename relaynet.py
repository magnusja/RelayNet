import torch
import torch.nn as nn
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, num_input_channels=1, kernel=(3, 7), stride=1, num_output_channels=64, dropout_prob=0.3):
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

        if dropout_prob > 0:
            self.model.add_module(str(len(self.model)), nn.Dropout2d(dropout_prob))

    def forward(self, input):
        return self.model(input)


class DenseBlock(nn.Module):
    def __init__(self, num_input_channels=1, kernel=(3, 7), stride=1, num_output_channels=64, dropout_prob=0.3):
        super().__init__()
        self.dense_modules = nn.ModuleList([
            BasicBlock(num_input_channels, kernel, stride, num_output_channels, dropout_prob),
            BasicBlock(num_input_channels + num_output_channels, kernel, stride, num_output_channels, dropout_prob),
            BasicBlock(num_input_channels + 2 * num_output_channels, (1, 1), stride, num_output_channels, dropout_prob)
        ])

    def forward(self, input):
        outputs = []
        for module in self.dense_modules:
            input_cat = torch.cat([input] + outputs, dim=1)
            output = module(input_cat)
            outputs.append(output)

        return outputs[-1]


class EncoderBlock(nn.Module):
    def __init__(self, num_input_channels=1, kernel=(3, 7), stride_conv=1, stride_pool=2, num_output_channels=64,
                 dropout_prob=0.3, basic_block=BasicBlock):
        super().__init__()
        self.basic = basic_block(num_input_channels, kernel, stride_conv, num_output_channels, dropout_prob)
        self.pool = nn.MaxPool2d(kernel, stride_pool, return_indices=True)

    def forward(self, input):
        tmp = self.basic(input)
        out, indices = self.pool(tmp)
        return out, indices, tmp


class DecoderBlock(nn.Module):
    def __init__(self, num_input_channels=64, kernel=(3, 7), stride_conv=1, stride_pool=2, num_output_channels=64,
                 dropout_prob=0.3, basic_block=BasicBlock):
        super().__init__()
        self.basic = basic_block(num_input_channels * 2, kernel, stride_conv, num_output_channels, dropout_prob)
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
                 num_encoders=3, num_classes=10, kernel_classify=(1, 1), dropout_prob=0.3, basic_block=BasicBlock):
        super().__init__()
        self.encoders = nn.ModuleList([EncoderBlock(num_input_channels if i == 0 else num_output_channels, kernel,
                                                    stride_conv, stride_pool, num_output_channels, dropout_prob)
                                       for i in range(num_encoders)])
        self.bottleneck = basic_block(num_output_channels, kernel, stride_conv, num_output_channels, dropout_prob)
        self.decoders = nn.ModuleList(
            [DecoderBlock(num_output_channels, kernel, stride_conv, stride_pool, num_output_channels, dropout_prob)
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

    def train(self, mode=True):
        super().train(mode)

        # to do MC dropout we would like to keep dropout also during evaluation
        for module in self.modules():
            if 'dropout' in  module.__class__.__name__.lower():
                module.train(False)

    def predict(self, input, times=10):
        self.eval()
        results = list()
        for _ in range(times):
            out = self.forward(input)
            results.append(out.data.cpu().numpy())

        results = np.asarray(results, dtype=np.float).squeeze()
        average = results.mean(axis=0)
        per_class_entropy = -np.sum(results * np.log(results), axis=0)
        overall_entropy = -np.sum(results * np.log(results), axis=(0, 1))

        return average, per_class_entropy, overall_entropy