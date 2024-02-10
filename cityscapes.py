#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transformation import *

class CityScapes(Dataset):
    def __init__(self, path, mode='train', cropsize=(1024, 512), 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        self.path =  "/content/Cityscapes/Cityspaces"                    #Path to Dataset
        assert mode in ('train', 'val')                                  #Dataset can be used for training and for validatio
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255                                             #To ignore void Class

        self.transformed_data = T.Compose([
          T.ToTensor(),
          T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ])
        
        self.trans_train = Compose([
           #ColorJitter(
               #brightness = 0.5,
               #contrast = 0.5,
               #saturation = 0.5),
           HorizontalFlip(),
           # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
           #RandomScale(randomscale),
           # RandomScale((0.125, 1)),
           # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
           # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
           RandomCrop(cropsize)
           ])

        with open('/content/DAAI_semantic-segmentation/cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        

        ## parse img directory
        self.imgs = {}                                #To store img path
        imgnames = []                                 #To store image names
        impth = osp.join(path, 'images', mode)        #To reconstruct the path of the image considering the mode
        folders = os.listdir(impth)                   #To list all the folders in the reconstructed path
        for fd in folders:                            #For each folder
            fdpth = osp.join(impth, fd)               #to create the path
            im_names = os.listdir(fdpth)              #to list all the images inside
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]    #To store inside names the filename
            impths = [osp.join(fdpth, el) for el in im_names]                  #To construct the full path for each image file till the current subdirectory
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(path, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelTrainIds' in el]
            names = [el.replace('_gtFine_labelTrainIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames                        #To assign the list of modified image filenames to self.imnames
        self.len = len(self.imnames)                   #To evaluate the lenght of the list 
        print('self.len', self.mode, self.len)
        assert set(imgnames) == set(gtnames)           #To ensure that the set in image is equal to the set of labels
        assert set(self.imnames) == set(self.imgs.keys())    #To ensure that all the file names have a corresponing path 
        assert set(self.imnames) == set(self.labels.keys())



    def __getitem__(self, idx):
        fn  = self.imnames[idx]                            #To retrieve the filename of the image at index idx
        impth = self.imgs[fn]                              #To reconstruct the full path of the image
        lbpth = self.labels[fn]                            #To reconstruct the full path of the label
        img = Image.open(impth).convert('RGB')             #To open the image as RGB
        label = Image.open(lbpth)                          #To open the label 
        if self.mode == 'train':                           #If Training 
            im_lb = dict(im = img, lb = label)             # To create a dictionary with the image and the label 
            im_lb = self.trans_train(im_lb)                #To apply transformation
            img, label = im_lb['im'], im_lb['lb']          #To update the image and the label with the transformed image and label 
        img = self.transformed_data(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]        #To convert label in NumPy array
        #label = self.convert_labels(label)
        return img, label


    def __len__(self):
        return self.len                                     #To return the lenght 


    #def convert_labels(self, label):
     #   for k, v in self.lb_map.items():
      #      label[label == k] = v
       # return label



if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))
