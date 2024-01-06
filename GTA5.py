import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import get_label_info_custom
from utils import from_label_to_TrainID
from torchvision import transforms

class GTA5Dataset(Dataset):
    def __init__(self, mode = 'train'):
        self.path = "/content/GTA5"
        self.mode = mode
        self.label_info = get_label_info_custom('/content/DAAI_semantic-segmentation/GTA5.csv')                  #I create the list with the info coming from the .csv
        self.images_dir = os.path.join(self.path, 'images/')                                                     #To load the path of the images
        self.labels_dir = os.path.join(self.path, 'labels/')                                                     #To load the path of the labels
        self.image_files = os.listdir(self.images_dir)                                                           #To load the pathe containg the names of the images
        self.label_colored_files = os.listdir(self.labels_dir)                                                   #To load the pathe containg the names of the labels
        #self.data, self.label_colored = self.data_loader()                                                 #To load the path of the image and the colored label 
        self.width = 1024
        self.height = 512
        self.transform_data = transforms.Compose([ 
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),                 
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.label = from_label_to_TrainID(self.label_colored_files,self.label_info, self.labels_dir, self.height, self.width)                                  # Convert label to TrainID format

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_files[idx])
        label_name = os.path.join(self.label[idx])
        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')
        
        
        tensor_image = self.transform_data(image)
        tensor_label = torch.from_numpy(np.array(label))  
        return tensor_image, tensor_label 


    #def data_loader(self):
    #    data= []
    #   label = []
    #    types = [ "labels/","images/"]
           
    #    for t in types:
    #        for root, dirs, files in os.walk(self.path+t):
    #            for file in files:
    #                file_path = os.path.join(root, file)
    #                relative_path = os.path.relpath(file_path, self.path)
    #                if t=="images/":
    #                    data.append(relative_path)
    #                else:
    #                    label.append(relative_path)
    #                if len(data)==len(label):
    #                    break
    #        return sorted(data), sorted(label)
            
