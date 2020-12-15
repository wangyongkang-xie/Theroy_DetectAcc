import random
from collections import Counter

import numpy as np
import keras
import torch
import torchvision
from keras.datasets import mnist
from keras import backend as K
import csv, glob, os, json

from torchvision.transforms import transforms


class DataSource(object):
    def __init__(self):
        raise NotImplementedError()

    def partitioned_by_rows(self, num_workers, test_reserve=0):
        raise NotImplementedError()

    def partitioned_by_columns(self, weight=None):
        raise NotImplementedError()


class CIFAR_10(DataSource):
    IID = True
    MAX_NUM_CLASSES_PER_CLIENT = 10
    def __init__(self, num=20, num_worker=10):
        self.transform_train1 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])
        self.transform_train2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])
        self.trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=self.transform_train1)
        self.testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=self.transform_train2)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=128, shuffle=False)
        self.group = [i for i in range(num)]
        random.shuffle(self.group)
        self.num_worker = num_worker
        self.num = num
        # self.trainset = trainset
        self.return_list = self.seperate_into_group()
        self.flag = [0 for i in range(self.num_worker)]
        self.data_index = [[] for i in range(self.num_worker)]
        for i in range(self.num_worker):
            self.data_index[i] += self.return_list[self.group[2 * i]] + self.return_list[self.group[2 * i + 1]]
            random.shuffle(self.data_index[i])

    def seperate_into_group(self):
        label_list = []
        for _, label in self.trainset:
            label_list.append(label)
        c = Counter(label_list)
        print(c)
        if self.num == 20:
            separate_ratio = 0.4
            return_list = [[] for i in range(self.num)]
            flag_list = [0 for i in range(10)]
            finish_list = [c[i] * separate_ratio for i in range(10)]
            print(finish_list)
            for i, label in enumerate(label_list):
                if flag_list[label] > finish_list[label]:
                    return_list[2 * label + 1].append(i)
                else:
                    return_list[2 * label].append(i)
                flag_list[label] += 1
        # print(return_list[0])
        return return_list

    def step(self, num, bz):
        data_list = []
        label_list = []
        for i in range(len(self.data_index[num])):
            data_list.append(self.trainset[self.data_index[num][self.flag[num]]][0])
            # label_list.append(self.trainset[np.array(self.data_index[num])][0])
            label_list.append(self.trainset[self.data_index[num][self.flag[num]]][1])
            self.flag[num] += 1
            self.shuffle()
        #        data = torch.stack(data_list,0)
        #        label = torch.tensor(label_list)
        # 	data_list = data_list.numpy()
        # 	label_list = data_list.numpy()
        return data_list, label_list

    def shuffle(self):
        for i, f in enumerate(self.flag):
            if f >= len(self.data_index[i]):
                random.shuffle(self.data_index[i])
                self.flag[i] = 0

    def client(self):
        self.client = []
        for i in range(len(self.data_index)):
            data_list, label_list = self.step(i, 0)
            self.client.append((data_list, label_list))
        return self.client
class Mnist(DataSource):
    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 10

    def __init__(self, num,num_worker):
        # self.classes = None
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self.x = x_train
        self.y = y_train
        self.x_test_global = x_test
        self.y_test_global = y_test
        self.classes = np.unique(self.y)
        n = self.x.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        self.x = self.x[idx]  # n * 28 * 28
        self.y = self.y[idx]  # n * 1
        # self.attack_mode = attack_mode
        self.group = [i for i in range(num)]
        random.shuffle(self.group)
        self.num_worker = num_worker
        self.num = num
        self.return_list = self.seperate_into_group()
        self.flag = [0 for i in range(self.num_worker)]
        self.data_index = [[] for i in range(self.num_worker)]
        for i in range(self.num_worker):
            self.data_index[i] += self.return_list[self.group[2 * i]] + self.return_list[self.group[2 * i + 1]]
            random.shuffle(self.data_index[i])

    def global_train(self):
        return self.post_process(self.x, self.y)

    def test_data(self):
        return self.post_process(self.x_test_global, self.y_test_global)

    def seperate_into_group(self):

        c = Counter(self.y)
        print(c)
        if self.num == 20:
            separate_ratio = 0.4
            self.return_list = [[] for i in range(self.num)]
            flag_list = [0 for i in range(10)]
            finish_list = [int(c[i] * separate_ratio) for i in range(10)]
            print(finish_list)
            for i, label in enumerate(self.y):
                if flag_list[label] > finish_list[label]:
                    self.return_list[2 * label + 1].append(i)
                else:
                    self.return_list[2 * label].append(i)
                flag_list[label] += 1
        return self.return_list

    def step(self, num):

        data_list = self.x[self.data_index[num]]
        label_list = self.y[self.data_index[num]]
        print('label_list:',Counter(label_list))
        return self.post_process(data_list, label_list)

    def shuffle(self):
        for i, f in enumerate(self.flag):
            if f >= len(self.data_index[i]):
                random.shuffle(self.data_index[i])
                self.flag[i] = 0

    def client(self):
        self.client = []
        for i in range(len(self.data_index)):
            data = self.step(i)
            self.client.append(data)
        return self.client

    def post_process(self, xi, yi):
        if len(xi.shape) == 2:
            xi = xi.reshape(1, xi.shape[0], xi.shape[1])
        if K.image_data_format() == 'channels_first':
            xi = xi.reshape(1, xi.shape[0], xi.shape[1], xi.shape[2])
        else:
            xi = xi.reshape(xi.shape[0], xi.shape[1], xi.shape[2], 1)

        y_vec = keras.utils.to_categorical(yi, self.classes.shape[0])
        # print('y_vec:', y_vec)
        # print('xi.shape:', xi.shape)
        return (xi / 255., y_vec)

    # split evenly into exact num_workers chunks, with test_reserve globally
    def partitioned_by_rows(self, num_workers, test_reserve=0):
        n_test = int(self.x.shape[0] * test_reserve)
        n_train = self.x.shape[0] - n_test
        nums = [n_train // num_workers] * num_workers
        nums[-1] += n_train % num_workers
        print(nums)
        idxs = np.array([np.random.choice(np.arange(n_train), num, replace=False) for num in nums])
        print(idxs)
        return {
            # (size_partition * 28 * 28, size_partition * 1) * num_partitions
            "train": [self.post_process(self.x[idx], self.y[idx]) for idx in idxs],
            # (n_test * 28 * 28, n_test * 1)
            "test": self.post_process(self.x[np.arange(n_train, n_train + n_test)],
                                      self.y[np.arange(n_train, n_train + n_test)])
        }
class FEMNIST(DataSource):

    NUM_CLASS = 62
    WIDTH = 28
    HEIGHT = 28

    def __init__(self, writer, attack_mode=0):

        self.writer = writer
        self.attack_mode = attack_mode

        self.x_train, self.y_train, self.x_test, self.y_test, self.x_valid, self.y_valid = self.read_data(writer)

        self.num_classes = FEMNIST.NUM_CLASS

        # self.x_test = self.x[num_train:num_train + num_test]
        # self.x_valid = self.x[num_train + num_test:]
        # self.y_train = self.y[0:num_train]
        # self.y_test = self.y[num_train:num_train + num_test]
        # self.y_valid = self.y[num_train + num_test:]


    def read_data(self, writer, data_split=(0.9, 0.1)):

        train_dir = "/data/app/v_wbsyili/Projects/leaf/data/femnist/data/femnist_lsy/train"
        test_dir  = "/data/app/v_wbsyili/Projects/leaf/data/femnist/data/femnist_lsy/test"
        # train_dir = "/root/Projects/Test/output_train"
        # test_dir  = "/root/Projects/Test/output_test"
        # train_dir = "/root/Projects/femnist_lsy/train"
        # test_dir  = "/root/Projects/femnist_lsy/test"

        # train_file = os.path.join(train_data_dir, writer)
        # test_file  = os.path.join(test_data_dir,  writer)
        train_file = os.path.join(train_dir, writer)
        test_file  = os.path.join(test_dir,  writer)

        with open(train_file, 'r') as fr:
            train_data = json.load(fr)
        with open(test_file, 'r') as fr:
            test_data = json.load(fr)
        valid_data = {}

        train_data["x"] = np.asarray( train_data["x"] )
        train_data["y"] = np.asarray( train_data["y"] )

        test_data["x"]  = np.asarray( test_data["x"] )
        test_data["y"]  = np.asarray( test_data["y"] )

        valid_size = int( (train_data["x"].shape[0] + test_data["x"].shape[0]) * data_split[1])
        valid_idx = np.random.choice( np.arange(train_data["x"].shape[0]),
                        size=valid_size, replace=False )
        train_idx = np.array( [ item for item in range(train_data["x"].shape[0])
                                    if item not in valid_idx ] )

        valid_data["x"] = train_data["x"][valid_idx]
        valid_data["y"] = train_data["y"][valid_idx]

        train_data["x"] = train_data["x"][train_idx]
        train_data["y"] = train_data["y"][train_idx]


        valid_data["x"] = np.reshape( valid_data["x"], (valid_data["x"].shape[0],
                                FEMNIST.WIDTH, FEMNIST.HEIGHT) )

        train_data["x"] = np.reshape( train_data["x"], (train_data["x"].shape[0],
                                FEMNIST.WIDTH, FEMNIST.HEIGHT) )

        test_data["x"]  = np.reshape( test_data["x"], (test_data["x"].shape[0],
                                FEMNIST.WIDTH, FEMNIST.HEIGHT) )

        return train_data["x"], train_data["y"], test_data["x"], test_data["y"], valid_data["x"], valid_data["y"]

    # assuming client server already agreed on data format
    def post_process(self, xi, yi):
        if len(xi.shape)==2:
            xi = xi.reshape(1, xi.shape[0], xi.shape[1])
        if K.image_data_format() == 'channels_first':
            xi = xi.reshape(1, xi.shape[0], xi.shape[1], xi.shape[2])
        else:
            xi = xi.reshape(xi.shape[0], xi.shape[1], xi.shape[2], 1)

        y_vec = keras.utils.to_categorical(yi, self.num_classes)
        return xi, y_vec



    def fake_non_iid_data(self):

        ## Wrong Label Attack
        print("Before Attack {}: {}".format(self.attack_mode, self.y_train[:10]))
        if self.attack_mode == 5:
            fake_label = np.random.randint(0, 61)
            self.y_train = np.full( self.y_train.shape, fake_label)
        print("After  Attack {}: {}".format(self.attack_mode, self.y_train[:10]))

        train_set = self.post_process(self.x_train, self.y_train)
        test_set  = self.post_process(self.x_test, self.y_test)
        valid_set = self.post_process(self.x_valid, self.y_valid)

        return ((train_set, test_set, valid_set), [])

