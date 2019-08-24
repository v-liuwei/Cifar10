import visdom
import time
import numpy as np


class Visualizer(object):
    '''
    封装了visdom的基本操作， 但仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    '''
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        :param env:
        :param kwargs:
        :return:
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        :param d: dict (name, value) i.e. ('loss', 0.11)
        :return:
        '''
        for k, v in d.item():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.item():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss', 0.11)
        :param name:
        :param y:
        :param kwargs:
        :return:
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img',t.Tensor(28,28))
        self.img('input_img',t.Tensor(3,28,28))
        self.img('input_img',t.Tensor(100,1,28,28))
        self.img('input_img',t.Tensor(100,3,28,28),nrows=10)
        don't self.img('input_img',t.Tensor(100,28,28),nrows=10)
        '''
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        :param info:
        :param win:
        :return:
        '''
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'),
                                                        info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        '''
        自定义的plot, image, log, plot_many等除外
        self.function等价于self.vis.function
        :param name:
        :return:
        '''
        return getattr(self.vis, name)
