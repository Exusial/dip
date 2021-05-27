import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_path='training',train=True):
        self.data_path = data_path
        self.dirs = os.listdir(data_path)
        self.train = train
        self.dataset = []

    def build_dataset(self):
        self.dataset = []
        for dir in self.dirs:
            path = os.path.join(self.data_path, dir)
            label = int(dir[:3:]) - 1
            print(label)
            img_names = os.listdir(path)
            target_img = np.random.randint(0,9)
            for num,name in enumerate(img_names):
                img_path = os.path.join(path, name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                img = (img - img.min()) / (img.max() - img.min())
                # print(img_path, img.shape)
                self.dataset.append((
                    torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1),
                    label,
                ))
        if not self.train:
            print('Building the validation dataset... data_size:', len(self.dataset))
        else:
            print('Building a new training dataset... data_size:', len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class FeatureDataset(Dataset):
    def __init__(self,data_source,idx,train=True):
        self.data_source = data_source
        self.idx = idx
        self.train = train
        self.dataset = []

    def build_dataset(self):
        self.dataset = []
        idx = self.idx
        for i in range(1000):
            for j in range(100):
                self.dataset.append((
                    torch.from_numpy(self.data_source[i*100+j]),
                    idx[i],
                ))  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
