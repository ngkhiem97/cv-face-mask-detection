import torch.nn as nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_planes, output_planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, output_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_planes)
        self.conv2 = nn.Conv2d(output_planes, output_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or input_planes != output_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_planes, output_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=2, init_weights=True):
        super(ResNet, self).__init__()
        self.input_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._create_layer(64, 2, stride=1)  
        self.layer2 = self._create_layer(128, 2, stride=2) 
        self.layer3 = self._create_layer(256, 2, stride=2) 
        self.layer4 = self._create_layer(512, 2, stride=2) 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _create_layer(self, output_planes, num_blocks, stride):
        layers = []
        for stride in [stride] + [1]*(num_blocks-1):
            layers.append(ResidualBlock(self.input_planes, output_planes, stride))
            self.input_planes = output_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)