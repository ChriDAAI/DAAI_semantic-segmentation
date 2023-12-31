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


class CityScapes(Dataset):
    def __init__(self, path, mode='train'):
        super(CityScapes, self).__init__()
        self.path =  "/content/Cityscapes/Cityspaces"    #To find the dataset
        assert mode in ('train', 'val')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        self.transformed_data = T.Compose([
          T.ToTensor(),
          T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ])

        with open('/content/DAAI_semantic-segmentation/cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(path, 'images', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
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
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())



    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.transformed_data(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return img, label


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))
