# python -m trial.build_dataset --dataset_name briareo --dataset_root_path data/Briareo
# Since we use relative import

import argparse
import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
import math
import numpy as np
import cv2
import torch
import torch.utils.data as data
from PIL import Image
# from spatial_transforms import *
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random
import glob
from torchvision import get_image_backend
import pdb
import pandas as pd
import ast
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image




# Relative import
from .generate_csv import generate_csv
from .build_transform import standard_transform

class DatasetImgTarget(data.Dataset):
    def __init__(self, root, split, transforms = None , n_frames = 40):
        # Split is train, test, val
        
        self.transforms = transforms
        self.root = root
        self.split = split
        self.images_folder = root
        self.n_frames = n_frames
        print("---self.root", self.root)
        path_csv = os.path.join(self.root, f'{split}.csv')
        self.df = pd.read_csv(path_csv, sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()
        
        # Should sampling be done within this class or not?
        # For example, what is another dataset have their own way of sampling?
        
        # Stores the path to images selected after sampling
        fixed_data = []
        for i, record in enumerate(self.data):
            # used ast because the array being passed record is in the form of a string
            # For more info, try printing type(record) or print len(record)
            record = ast.literal_eval(record)
            center_of_list = math.floor(len(record)/2)
            crop_limit = math.floor(self.n_frames / 2)
            start = center_of_list - crop_limit
            end = center_of_list + crop_limit 
            # Add one more extra frame if n_frames is odd  
            paths_cropped = record[start: end + 1 if self.n_frames % 2 == 1 else end + 1]
            # Adding arrays of cropped clips for every video_sample
            fixed_data.append(paths_cropped)
        
        self.data = fixed_data
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        paths = self.data[index]
        label = self.targets[index]
        
        clip = []
        seed = np.random.randint(2147483647)  # Generate one seed for the entire clip
        
        # Process each frame in a gesture clip
        for p in paths:
            # p[0] because apparently p is a tuple
            path_frame = os.path.join(self.root,  p[0])
            # img = cv2.imread(path_frame, cv2.IMREAD_COLOR)
            # # cv2.imshow("image", img)
            # # cv2.waitKey(0)
            # # cv2.destroyAllWindows()
            # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # rgb_img = cv2.resize(rgb_img, (224, 224))
            # clip.append(rgb_img)
            

            # Set seed for PyTorch random number generator
            torch.manual_seed(seed)

            
            # H, W, C
            img = Image.open(path_frame)
            rgb_img = img.convert("RGB")
            rgb_img = self.transforms(rgb_img)
            clip.append(rgb_img)
        
        # Passing it to transformation function makes it (c, h, w) by default)
        # If we did not pass it through the transformation function, we have to transpose it since it's (h, w, c) if I'm not mistaken
        clip = np.array(clip)

           

        clip = torch.from_numpy(clip)
        # print("*****", clip.shape)
        label = torch.LongTensor(np.asarray([label]))
        
        return clip.float(), label

def vis_dataset(args):
    transform_function = standard_transform(args, True)
    # print("Testing", transform_function)
    result_dir = r"dataset_generation\\tester_images"
    train_set = DataLoader(DatasetImgTarget(root=args.dataset_root_path, split='train', transforms=transform_function), batch_size=1)
    test_set = DataLoader(DatasetImgTarget(root=args.dataset_root_path, split='test', transforms=transform_function), batch_size=1)
    val_set = DataLoader(DatasetImgTarget(root=args.dataset_root_path, split='val'), batch_size=1)
    for split, loader in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
        for idx, (images, _) in enumerate(loader):
            print("Shape of images is ", images.shape)
            # (B, F, C, H, W)
            batch_size, frames, channels, height, width = images.shape
            channels = 3
            # (batch, frames, channel, height, width)
            # images = images.permute(0, 1, 4, 2, 3)
            
            # Convention is B,F,C,H,W
            images = images.view(-1, channels, height, width)
            
            # images = images.view(batch_size * frame, channels, height, width )
            print("Revised image shape is ", images.shape)
            
            # Why add this line? Why must it be divided by 255?
            # Add only to normalize when not passing it through a transformation function
            # images = images.float() / 255.0
            
            fp = os.path.join(result_dir, split, f'{idx}.png')
            save_image(images, fp, nrow=int(math.sqrt(images.shape[0])))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required = True, help="path to dataset root")
    parser.add_argument('--dataset_name', type=str, required = True, help = "name of the dataset")
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--random_erasing', action="store_true", help = "trigger random erasing operations")
    parser.add_argument('--random_horizontal_flip', action="store_true")
    parser.add_argument('--random_cropping', action="store_true")
    parser.add_argument('--restricted_rotation', action="store_true")
    parser.add_argument('--shear', action="store_true")
    parser.add_argument('--translate', action="store_true")
    args = parser.parse_args()
    

    if(args.dataset_name =='briareo'):
        generate_csv(args.dataset_root_path, split='train')
        generate_csv(args.dataset_root_path, split='val')
        generate_csv(args.dataset_root_path, split='test')
        
        if(args.visualize):
            vis_dataset(args)
        else:
            print("not visualize")
    


if __name__ == '__main__':
    main()