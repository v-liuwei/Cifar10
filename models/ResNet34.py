from torch import nn
from torch.nn import functional as F
from models import BasicModule


class ResidualBlock(BasicModule):
    '''
    实现子module：ResidualBlock
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = out+residual
        return F.relu(out)


class ResNet34(BasicModule):
    '''
    实现主module：ResNet34
    ResNet34包含多个layer， 每个layer又包含多个residual block
    用子module实现residual block， 用_make_layer函数实现layer
    '''
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # N*3*32*32->N*64*16*16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # N*64*16*16->N*64*8*8
        )
        self.layer1 = self._make_layer(64, 64, 3)  # N*64*8*8->N*64*8*8
        self.layer2 = self._make_layer(64, 128, 4, stride=2)  # N*64*8*8->N*128*4*4
        self.layer3 = self._make_layer(128, 256, 6, stride=2)  # N*128*4*4->N*256*2*2
        self.layer4 = self._make_layer(256, 512, 3, stride=2)  # N*256*2*2->N*512*1*1

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        构建layer， 包含多个residual block
        :param inchannel:
        :param outchannel:
        :param block_num:
        :param stride:
        :return:
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)  # N*3*32*32->N*64*8*8

        x = self.layer1(x)  # N*64*8*8->N*64*8*8
        x = self.layer2(x)  # N*64*8*8->N*128*4*4
        x = self.layer3(x)  # N*128*4*4->N*256*2*2
        x = self.layer4(x)  # N*256*2*2->N*512*1*1

        x = F.adaptive_avg_pool2d(x, 1)  # N*512*1*1->N*512*1*1
        x = x.view(x.size(0), -1)  # N*512*1*1->N*512
        return self.fc(x)  # N*512->N*10
