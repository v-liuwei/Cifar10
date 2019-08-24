import warnings
import torch as t
import torch.nn as nn


class DefaultConfig(object):
    env = 'default'
    model = 'ResNet34'

    train_data_root = 'F:\\Machine_Learning\\Datasets\\cifar10\\train.pkl'
    test_data_root = 'F:\\Machine_Learning\\Datasets\\cifar10\\test.pkl'
    load_model_path = None
    save_model_path = 'F:\\Machine_Learning\\Projects\\cifar10_classify\\checkpoints\\model'

    batch_size = 50
    use_gpu = True
    num_workers = 1
    print_freq = 20

    result_file = 'result.csv'

    max_epoch = 8
    lr = 1e-3
    lr_decay = 0.95
    weight_decay = 1e-4
    criterion = nn.MSELoss()
    optimizer = t.optim.Adam

    def parse(self, **kwargs):
        '''
        根据字典更新config参数
        :param kwargs:
        :return:
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has not attribute %s' % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
