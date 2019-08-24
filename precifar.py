import os
import pickle
import numpy as np


def unpickle(file):
    dict = pickle.load(open(file, 'rb'), encoding='bytes')
    return dict


def get_photo(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024].reshape(1, 32, 32)
    g = pixel[1024:2048].reshape(1, 32, 32)
    b = pixel[2048:3072].reshape(1, 32, 32)
    photo = np.concatenate([r, g, b], 0)
    return photo


def GetTrainData():
    data = []
    labels = []
    batch_label = []
    filenames = []
    for i in range(1, 6):
        dict = unpickle('F:\\Machine_Learning\\Datasets\\cifar10\\data_batch_%d' % i)
        filenames.append(dict[b'filenames'])
        batch_label.append(dict[b'batch_label'])
        data.append(dict[b'data'])
        labels.append(dict[b'labels'])
    data = np.concatenate(data, 0)
    labels = np.concatenate(labels, 0)
    return [data, labels], batch_label, filenames


def GetTestData():
    dict = unpickle('F:\\Machine_Learning\\Datasets\\cifar10\\test_batch')
    data = dict[b'data']
    labels = np.array(dict[b'labels'])
    batch_label = dict[b'batch_label']
    filenames = dict[b'filenames']
    return [data, labels], batch_label, filenames


def GetLabelArray(Labels):
    result = np.zeros([Labels.shape[0], 10])
    for i in range(result.shape[0]):
        result[i][int(Labels[i])] = 1
    return result


def Precifar():
    Train_Data, tra_batch_label, tra_filenames = GetTrainData()
    Test_Data, test_batch_label, test_filenames = GetTestData()
    Train_Data[1] = GetLabelArray(Train_Data[1])
    Test_Data[1] = GetLabelArray(Test_Data[1])
    Train_Data[0] = Train_Data[0].reshape((-1, 3, 32, 32))/255
    Test_Data[0] = Test_Data[0].reshape((-1, 3, 32, 32))/255
    Train_Data = tuple(Train_Data)
    Test_Data = tuple(Test_Data)
    pickle.dump(Train_Data, open('F:\\Machine_Learning\\Datasets\\cifar10\\train.pkl', 'wb'))
    pickle.dump(Test_Data, open('F:\\Machine_Learning\\Datasets\\cifar10\\test.pkl', 'wb'))


if __name__ == "__main__":
    Precifar()
