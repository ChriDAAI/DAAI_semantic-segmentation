import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class CityScapes(Dataset):
    def __init__(self, mode):
        self.path = "/content/Cityscapes/Cityspaces"
        self.mode = mode
        self.images = os.path.join(self.path, 'images/', mode)
        self.gtFine = os.path.join(self.path, 'gtFine/', mode)
        self.list, self.labels = self.getdata()
        self.width = 1024
        self.height = 512
        self.transform_data = transforms.Compose([ 
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),                 
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.list[idx])
        label_name = os.path.join(self.labels[idx])
        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')
        
        
        tensor_image = self.transform_data(image)
        tensor_label = torch.from_numpy(np.array(label))  
        return tensor_image, tensor_label 
      
    def getdata(self):
      list = []
      labels = []
    
      for city in os.listdir(self.images):
        img_city_dir = os.path.join(self.images, city)
        label_city_dir = os.path.join(self.gtFine, city)
        if os.path.isdir(img_city_dir) and os.path.isdir(label_city_dir):
          for img_name in os.listdir(img_city_dir):
            label_name = img_name.replace("_leftImg8bit.png", "_labelTrainIds.png")
            img_path = os.path.join(img_city_dir, img_name)
            list.append(img_path)
            label_path = os.path.join(label_city_dir, label_name)
            label.append(label_path)
            
      return sorted(list), sorted(label)
