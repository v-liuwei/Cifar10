from torch.utils import data
import torch as t
import pickle


class Cifar10(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        '''
        获取所有图片地址，并根据训练、验证、测试划分数据
        :param root:
        :param transforms:
        :param train:
        :param test:
        '''
        self.test = test
        self.train = train
        data = pickle.load(open(root, 'rb'))
        if self.test:
            self.data = data
        elif self.train:
            self.data = (data[0][:int(0.8*len(data[0]))], data[1][:int(0.8*len(data[0]))])
        else:
            self.data = (data[0][int(0.8*len(data[0])):], data[1][int(0.8*len(data[0])):])
        if transforms is None:
            pass

    def __getitem__(self, index):
        data = t.Tensor(self.data[0][index])
        label = t.Tensor(self.data[1][index])
        return data, label

    def __len__(self):
        return len(self.data[0])
