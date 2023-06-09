#Code written by Rakshith Sathish
#The work is made public with MIT License

import os
import collections
import torch
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from torch.utils import data


class lunaLoader(data.Dataset):
    def __init__(self,split="train",is_transform=True,img_size=512):
        self.split = split
        self.path= "/LUNA16/2ddataset/"+self.split
        self.is_transform = is_transform
        self.img_size = img_size
        self.files = os.listdir(self.path+'/images/') # [image1_img.npy, image2_img.npy]
        
        self.img_tf = transforms.Compose(
            [   transforms.Resize(self.img_size),
                transforms.ToTensor(),
                # transforms.Normalize([-460.466],[444.421])               
            ])
        
        self.label_tf = transforms.Compose(
            [   
            	transforms.Resize(self.img_size,interpolation=0),
                transforms.ToTensor(),
            ])
        
    
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,index):
        fname = self.files[index] # image1_img.npy, image1_label.npy
        img = Image.fromarray(np.load(self.path+'/images/'+fname).astype(float))
        im_id = fname.split('_')[1]
        label = Image.fromarray(np.load(self.path+'/labels/masks_'+im_id))
        
        if self.is_transform:
            img, label = self.transform(img,label)

        if self.split is 'train':
            img, label = self.aug_data(img,label)
        
        return img, label.squeeze(0)

    def aug_data(self,img,label):
        # Random horizontal flip
        if np.random.random() < 0.5:
            img = TF.hflip(img)
            label = TF.hflip(label)

        # Random vertical flip
        if np.random.random() < 0.5:
            img = TF.vflip(img)
            label = TF.vflip(label)

        # Random rotation
        angle = np.random.uniform(-10, 10)
        img = TF.rotate(img, angle)
        label = TF.rotate(label, angle)
        return img,label

    
    def transform(self,img,label):
        img = normalize(img)
        # img = zero_center(img)
        img = self.img_tf(img)
        label = self.label_tf(label)
        
        return img,label

def normalize(image):
    MIN_BOUND = -1200
    MAX_BOUND = 600.
    image = np.array(image, dtype=np.float32)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image *= 255.
    PIXEL_MEAN = 0.25 * 256
    image = image - PIXEL_MEAN
    image = image.astype(np.float32)
    image = Image.fromarray(image)
    return image

# def zero_center(image):
#     PIXEL_MEAN = 0.25 * 256
#     image = image - PIXEL_MEAN
#     image = Image.fromarray(image)