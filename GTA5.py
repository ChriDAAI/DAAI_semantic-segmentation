#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from collections import namedtuple
from pprint import pprint
from PIL import Image
import numpy as np
import pandas as pd
from utils import get_label_info_custom
from utils import from_label_to_TrainID


class GTA5Dataset(Dataset):
    def __init__(self, mode):
        super(GTA5Dataset, self).__init__()
        self.path = "/content/GTA5/"
        self.mode = mode
        self.label_info = get_label_info_custom('/content/DAAI_semantic-segmentation/GTA5.csv')                  #I create the list with the info coming from the .csv
        self.images_dir = os.path.join(self.path, 'images/')                                                     #To load the path of the images (/content/GTA5/images)
        self.labels_dir_colored = os.path.join(self.path, 'labels/')                                             #To load the path of the labels (/content/GTA5/labels)
        self.labels_dir_trainID = os.path.join(self.path, 'TrainID/')                                            #To load the path of the labels (/content/GTA5/TrainID)
        self.images_files, self.label_colored = self.data_loader()                                               #To have 'images/0000x.png' and 'labels/0000x.png'
        self.label_colored_files = self.label_colored.split("/")[-1]
        self.transform_data = transforms.Compose([ 
            transforms.ToTensor(),                 # Converte l'immagine in un tensore
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.width = 1024
        self.height = 512
        self.label=from_label_to_TrainID(self.label_colored_files,self.label_info, self.labels_dir_colored, self.height, self.width)
        #self.data_augmentation= DataAugmentation()
        #self.enable_da = enable_da
    
    def pil_loader(self, p, mode):
        with open(self.path+p, 'rb') as f:
            img = Image.open(f)
            return img.convert(mode).resize((self.width, self.height), Image.NEAREST)

    def __getitem__(self, idx):
        image = self.pil_loader(self.images_files[idx], 'RGB')
        label = self.pil_loader(self.label[idx], 'L')
        
        tensor_image = self.transform_data(image)                                                               #To have a tensor
        tensor_label = torch.from_numpy(np.array(label))                                                        #To have a tensor
        return tensor_image, tensor_label

    def __len__(self):
        return len(self.images_files)
    
    def data_loader(self):
        img= []
        lbl = []
        domain = [ "labels/","images/"]
        
        for d in domain:
            for root, dirs, files in os.walk(self.path+d):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.path)
                    if d=="images/":
                        img.append(os.path.basename(relative_path))
                    else:
                        lbl.append(os.path.basename(relative_path))
                    if len(img)==len(lbl):
                        break
        return sorted(img), sorted(lbl)
           
        
