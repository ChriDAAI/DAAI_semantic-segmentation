import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import get_label_info_custom
from utils import from_label_to_TrainID
from torchvision import transforms
import torch

class GTA5Dataset(Dataset):
    def __init__(self, mode):
        super(GTA5Dataset, self).__init__()
        self.path = "/content/GTA5"                                                                              #Main GTA5 directory 
        self.mode = mode                                                                                         #Train or validation
        self.label_info = get_label_info_custom('/content/DAAI_semantic-segmentation/GTA5.csv')                  #I create the list with the info coming from the .csv
        self.images_dir = os.path.join(self.path, 'images/')                                                     #To load the path of the images (/content/GTA5/images)
        self.labels_dir_colored = os.path.join(self.path, 'labels/')                                             #To load the path of the labels (/content/GTA5/labels)
        self.labels_dir_trainID = os.path.join(self.path, 'TrainID/')                                            #To load the path of the labels (/content/GTA5/TrainID)
        self.image_files, self.label_colored_files = self.img_loader()                                                   #To load the path containg the names of the images ('00001.png')
        self.width = 1024                                                                                        
        self.height = 512
        self.transform_data = transforms.Compose([ 
            transforms.ToTensor(),                 
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.label = from_label_to_TrainID(self.label_colored_files,self.label_info, self.labels_dir_colored, self.height, self.width)           #Convert label to TrainID format

    def __len__(self):
        return len(self.image_files)                                            

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_files[idx])                                          
        label_name = os.path.join(self.label[idx])                                                    
        with open(img_name, 'rb') as f: 
            image = Image.open(f).convert('RGB').resize((self.width, self.height), Image.NEAREST)                #I open the image, resize and convert in RGB
        with open(label_name, 'rb') as b:
            labels = Image.open(label_name).convert('L').resize((self.width, self.height), Image.NEAREST)        #I open the TrainID, resize and convert in L
        
        tensor_image = self.transform_data(image)                                                               #To have a tensor
        tensor_label = torch.from_numpy(np.array(labels))                                                        #To have a tensor
        return tensor_image, tensor_label
            
    def img_loader(self):
        img = []
        lbs = []
        domain = ["labels/", "images/"]
        for d in domain:
            for root, dirs, files in os.walk(self.path+d):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.path)
                    if d == "images/":
                        img.append(relative_path)
                    else:
                        lbs.append(relative_path)
                    if len(img)==len(lbs):
                        break
        return sorted(img), sorted(lbs)
