import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import get_label_info
from utils import one_hot_it
from torchvision import transforms

class GTA5Dataset(Dataset):
    def __init__(self, path, csv_file, transform, mode = 'train'):
        self.path = "/content/GTA5"
        self.mode = mode
        self.label_info = self.get_label_info('/content/DAAI_semantic-segmentation/GTA5.csv')
        self.images_dir = os.path.join(path, 'images')
        self.labels_dir = os.path.join(path, 'labels')
        self.image_files = os.listdir(self.images_dir)
        self.width = 1024
        self.height = 512
        self.transform_data = transforms.Compose([ 
            transforms.Resize((self.height, self.width))
            transforms.ToTensor(),                 
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        label_name = os.path.join(self.labels_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        label_colored = Image.open(label_name)
        label = self.one_hot_it(label_colored,self.label_info)  # Convert label to TrainID format
        
        return sorted(image), sorted(label)
