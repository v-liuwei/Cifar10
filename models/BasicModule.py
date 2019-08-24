import torch.nn as nn
import torch as t
import time


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        '''
        可加载指定路径的模型
        :param path:
        :return:
        '''
        self.load_state_dict(t.load(path))

    def save(self, path, name=None):
        if name is None:
            name = path+time.strftime('_%m%d_%H_%M.pkl')
        t.save(self.state_dict(), name)
        return name
