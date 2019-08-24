from models import BasicModule
import torch.nn as nn


class CNN(BasicModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=30,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                30, 40, 5, 1, 2
            ),
            # nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                40, 80, 4, 2, 1
            ),
            # nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Linear(80*2*2, 80)
        self.out = nn.Linear(80, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)               # 10*16*16
        x = self.conv2(x)               # 20*8*8
        x = self.conv3(x)               # 2*2
        x = x.view(x.size(0), -1)
        x = self.linear(x)              # 100
        x = self.out(x)                 # 10
        return x
