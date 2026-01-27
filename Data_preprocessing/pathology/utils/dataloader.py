# dataset and dataloader for pretraining
from torchvision import transforms
from torchvision.utils import save_image
import torch.utils.data as data
import numpy as np
from einops import rearrange
from PIL import ImageEnhance, Image, ImageOps
from skimage.color import rgb2hed, hed2rgb
import random
import os
from glob import glob
import torch
import logging

def random_flip(img):
    flip_flag = random.randint(0, 2)
    if flip_flag == 2:
        img = ImageOps.mirror(img)
    return img

def randomRotation(image):
    rotate_time = random.randint(0, 3)
    image = image.rotate(rotate_time * 90)
    return image

def colorEnhance(image):
    bright_intensity = random.randint(4, 16) / 10.0
    contrast_intensity = random.randint(4, 16) / 10.0
    color_intensity = random.randint(4, 16) / 10.0
    sharp_intensity = random.randint(4, 16) / 10.0
    
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    image = ImageEnhance.Color(image).enhance(color_intensity)
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(img, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return img


def randomPeper(img):
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 1
    return img

class Train_Dataset(data.Dataset):
    def __init__(self, opt):
        self.data_root = opt.data_root
        self.aug = opt.augment
        self.non_cancer = glob(self.data_root + '**/non_cancer/**/*.png',recursive=True)
        self.cancer = glob(self.data_root + '**/cancer/**/*.png', recursive=True)
        print("[Training set Stats:] [Non_cancerous Patches Num: %d] [Cancerous Patches Num: %d]"
              % (len(self.non_cancer), len(self.cancer)))
        self.img_path = self.non_cancer + 2 * self.cancer
        self.label = [0] * len(self.non_cancer) + [1] * 2 * len(self.cancer)
        print("[Training set Stats:] [Non_cancerous Patches Num: %d] [Cancerous Patches Num: %d]"
              % (len(self.non_cancer), 2 * len(self.cancer)))
        print('num samples:', len(self.img_path))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(opt.img_size)
        ])

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        label = self.label[index]
        if self.aug == False:
            img = self.img_transform(img)
        else:
            img = random_flip(img)
            img = randomRotation(img)
            img = colorEnhance(img)
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return len(self.img_path)

def get_dataloader(batch_size, shuffle, pin_memory, num_workers, opt):
    dataset = Train_Dataset(opt)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    return dataset, data_loader