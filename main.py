from config import DefaultConfig
from data import Cifar10
import models
from Trainer import Trainer
import csv
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms


opt = DefaultConfig()

if not os.path.exists(opt.train_data_root):
    from precifar import *
    Precifar()

train_data = Cifar10(opt.train_data_root, train=True)
val_data = Cifar10(opt.train_data_root, train=False)
test_data = Cifar10(opt.test_data_root, test=True)

Model = getattr(models, opt.model)()
if opt.load_model_path:
    Model.load(opt.load_model_path)
Cifar_Trainer = Trainer(Model, opt)

Cifar_Trainer.train(train_data, val_data)
Model.save(opt.save_model_path)

results, confusion_matrix, accuracy = Cifar_Trainer.test(test_data)

with open(opt.result_file, 'wt', newline='', encoding='utf-8') as csvfile:
    try:
        writer = csv.writer(csvfile)
        writer.writerow(("target", "predict"))
        for row in results:
            writer.writerow(row)
    finally:
        csvfile.close()
plt.imshow(confusion_matrix.value())
plt.show()
print("test accuracy:{}".format(accuracy))
