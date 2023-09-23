import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from models.utils import read


def random_resize(img0, imgt, img1, p=0.3):
    prob = random.uniform(0, 1)
    if 0 <= prob < p / 2:
        img0 = cv2.resize(img0, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    elif p / 2 <= prob < p:
        img0 = cv2.resize(img0, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return img0, imgt, img1

def random_crop(img0, imgt, img1, crop_size=(256, 256)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[x:x+h, y:y+w, :]
    imgt = imgt[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    return img0, imgt, img1

def random_reverse_channel(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]
    return img0, imgt, img1

def random_vertical_flip(img0, imgt, img1, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
    return img0, imgt, img1

def random_horizontal_flip(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
    return img0, imgt, img1

def random_rotate(img0, imgt, img1, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
    return img0, imgt, img1

def random_reverse_time(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img1
        img1 = img0
        img0 = tmp
    return img0, imgt, img1


class Vimeo90K_Train_Dataset(Dataset):
    def __init__(self, dataset_dir, augment=True):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        with open(os.path.join(dataset_dir, 'tri_trainlist.txt'), 'r') as f:
            for i in f:
                name = str(i).strip()
                if(len(name) <= 1):
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])

        if self.augment == True:
            img0, imgt, img1 = random_resize(img0, imgt, img1, 0.3)
            img0, imgt, img1 = random_crop(img0, imgt, img1, (256, 256))
            img0, imgt, img1 = random_reverse_channel(img0, imgt, img1, p=0.5)
            img0, imgt, img1 = random_vertical_flip(img0, imgt, img1, p=0.3)
            img0, imgt, img1 = random_horizontal_flip(img0, imgt, img1, p=0.5)
            img0, imgt, img1 = random_rotate(img0, imgt, img1, p=0.05)
            img0, imgt, img1 = random_reverse_time(img0, imgt, img1, p=0.5)

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)

        return img0, imgt, img1


class Vimeo90K_Test_Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        with open(os.path.join(dataset_dir, 'tri_testlist.txt'), 'r') as f:
            for i in f:
                name = str(i).strip()
                if(len(name) <= 1):
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)

        return img0, imgt, img1
