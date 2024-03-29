from torch.utils.data import Dataset, DataLoader
import torch
# from PIL import Image
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
import os


class dataset2(Dataset):

    def __init__(self, root_dir, dataset_path, transforms):
        self.root_dir = root_dir
        self.dataset = pd.read_csv(dataset_path)
        
        self.imgs = self.dataset.image_path.values
        self.labels = self.dataset.label.values
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(os.path.join(self.root_dir, self.imgs[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(self.imgs[idx])
            idx += 1
            image = cv2.imread(os.path.join(self.root_dir, self.imgs[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        if self.transforms:
            tr_img = self.transforms(image=image)
            image = tr_img['image']
        else:
            image = ToTensorV2(image)
        label = torch.tensor(label).float()

        return image, label

class dataset2_v2(Dataset):

    def __init__(self, dataset_path, transforms):
        self.dataset = pd.read_csv(dataset_path)
        self.imgs = self.dataset.image_path.values
        self.labels = self.dataset.label.values
        self.transforms = transforms
        self.source = self.dataset.source.values


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.imgs[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(self.imgs[idx])
            idx += 1
            image = cv2.imread(self.imgs[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        source = self.source[idx]
        id = self.imgs[idx]
        if self.transforms:
            tr_img = self.transforms(image=image)
            image = tr_img['image']
        else:
            image = ToTensorV2(image)
        label = torch.tensor(label).float()
        source = torch.tensor(source).float()


        return image, label, source, str(id)

class dataset2_v22(Dataset):

    def __init__(self, dataset_path, transforms):
        self.dataset = pd.read_csv(dataset_path)
        self.imgs = self.dataset.image_path.values
        self.labels = self.dataset.label.values
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.imgs[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(self.imgs[idx])
            idx += 1
            image = cv2.imread(self.imgs[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        if self.transforms:
            tr_img = self.transforms(image=image)
            image = tr_img['image']
        else:
            image = ToTensorV2(image)
        label = torch.tensor(label).float()

        return image, label
