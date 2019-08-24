from config import DefaultConfig
import torch as T
from torchnet import meter
from utils import Visualizer
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, model=None, opt=DefaultConfig()):
        self.model = model
        self.criterion = opt.criterion
        self.optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.opt = opt

    def train(self, train_data, val_data=None):
        print('Now begin training')
        train_dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True)
        vis = Visualizer(env=self.opt.env)

        if self.opt.use_gpu:
            self.model.cuda()

        previous_loss = 1e10
        loss_meter = meter.AverageValueMeter()
        confusion_matrix = meter.ConfusionMeter(10)

        for epoch in range(self.opt.max_epoch):
            loss_meter.reset()
            confusion_matrix.reset()
            for i, (data, label) in enumerate(train_dataloader):
                if self.opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda()
                self.optimizer.zero_grad()
                score = self.model(data)
                outclasses = T.argmax(score, 1)
                target_digit = T.argmax(label, 1)
                loss = self.criterion(score, label)
                loss.backward()
                self.optimizer.step()

                loss_meter.add(loss.data.cpu())
                confusion_matrix.add(outclasses, target_digit)
                accuracy = 100*sum(confusion_matrix.value()[i, i] for i in range(10)
                                   )/confusion_matrix.value().sum()
                if i % self.opt.print_freq == self.opt.print_freq-1:
                    print('EPOCH:{0},i:{1},loss:%.6f'.format(epoch, i) % loss.data.cpu())
                vis.plot('loss', loss_meter.value()[0])
                vis.plot('accuracy', accuracy)
                vis.img('Train Confusion_matrix', T.Tensor(confusion_matrix.value()))
            if val_data:
                val_cm, val_acc = self.test(val_data, val=True)
                vis.plot('val_accuracy', val_acc)
                vis.img('Val Confusion_matrix', T.Tensor(val_cm.value()))

            if loss_meter.value()[-1] > previous_loss:
                self.opt.lr *= self.opt.lr_decay
                print("learning rate:{}".format(self.opt.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.opt.lr
            previous_loss = loss_meter.value()[-1]

    def test(self, test_data, val=False):
        self.model.eval()
        confusion_matrix = meter.ConfusionMeter(10)
        test_dataloader = DataLoader(test_data, 2000, shuffle=True)

        results = []
        for i, (data, label) in enumerate(test_dataloader):
            if self.opt.use_gpu:
                data = data.cuda()
                label = label.cuda()
            score = self.model(data)
            out_digit = T.argmax(score, 1)
            target_digit = T.argmax(label, 1)
            if not val:
                bacth_results = [(target_digit.data.cpu().numpy(), out_digit.data.cpu().numpy())
                                 for target_digit, out_digit in zip(target_digit, out_digit)]
                results += bacth_results
            confusion_matrix.add(out_digit, target_digit)

        accuracy = 100*sum([confusion_matrix.value()[i][i] for i in range(10)])/confusion_matrix.value().sum()

        self.model.train()
        if val:
            return confusion_matrix, accuracy
        else:
            return results, confusion_matrix, accuracy
