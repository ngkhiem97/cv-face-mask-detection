import torch.nn as nn
import torch
class ConvBatchNormReLU(nn.Sequential):
    """
    Combination of Conv2d, BatchNorm2d and ReLU6.
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBatchNormReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.shortcut = stride == 1 and in_channel == out_channel
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBatchNormReLU(in_channel, hidden_channel, kernel_size=1)) # 1x1 pointwise conv / exand
        layers.append(ConvBatchNormReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel)) # 3x3 depthwise conv
        layers.append(nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False)) # 1x1 pointwise conv(linear)
        layers.append(nn.BatchNorm2d(out_channel))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.shortcut else self.conv(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes=2, init_weights=True):
        super(MobileNet, self).__init__()
        # t: expansion ratio
        # c: output channel
        # n: repeat times
        # s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = 32
        output_channel = 1280
        features = []
        features.append(ConvBatchNormReLU(3, input_channel, stride=2))
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBatchNormReLU(input_channel, output_channel, 1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(output_channel, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)    
