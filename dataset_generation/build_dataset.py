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



# Relative import
from .generate_csv import generate_csv

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
        
        # Process each frame in a gesture clip
        for p in paths:
            # p[0] because apparently p is a tuple
            path_frame = os.path.join(self.root,  p[0])
            img = cv2.imread(path_frame, cv2.IMREAD_COLOR)
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = cv2.resize(img, (224, 224))
            clip.append(img)
        
        clip = np.array(clip).transpose(1, 2, 3, 0)
        # print("_____", clip.shape)
        
        if self.transforms is not None:
            aug_det = self.transforms.to_deterministic()
            clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)

    
        # Clip looked like this (224, 224, 3, 41) 
        # the -1 at the end automatically computes the last dimension (infer?)
        clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
        # print("*****", clip.shape)
        label = torch.LongTensor(np.asarray([label]))
        
        return clip.float(), label

def vis_dataset(args):
    result_dir = r"dataset_generation\\visualized_images"
    train_set = DataLoader(DatasetImgTarget(root=args.dataset_root_path, split='train'), batch_size=1)
    test_set = DataLoader(DatasetImgTarget(root=args.dataset_root_path, split='test'))
    val_set = DataLoader(DatasetImgTarget(root=args.dataset_root_path, split='val'))
    for split, loader in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
        for idx, (images, _) in enumerate(loader):
            print("Shape of images is ", images.shape)
            batch_size, channel_times_frame, height, width = images.shape
            channels = 3
            frame = channel_times_frame // channels
            
            # Convention is B,F,C,H,W
            images = images.view(batch_size, frame, channels, height, width)
            
            images = images.view(batch_size * frame, channels, height, width )
            print("Revised image shape is ", images.shape)
            
            # Why add this line? Why must it be divided by 255?
            images = images.float() / 255.0
            
            fp = os.path.join(result_dir, split, f'{idx}.png')
            save_image(images, fp, nrow=int(math.sqrt(images.shape[0])))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required = True, help="path to dataset root")
    parser.add_argument('--dataset_name', type=str, required = True, help = "name of the dataset")
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()
    

    if(args.dataset_name =='briareo'):
        generate_csv(args.dataset_root_path, split='train')
        generate_csv(args.dataset_root_path, split='val')
        generate_csv(args.dataset_root_path, split='test')
        
        if(args.visualize):
            vis_dataset(args)
        else:
            print("not visualize")
    
        
        # train_set = DatasetImgTarget(root=args.dataset_root_path, split='train')
        # print(len(train_set))
        # for index, data in enumerate(DataLoader(train_set)):
        #     print(index)
        #     print(data)
        
    # vis_dataset(args)
        

if __name__ == '__main__':
    main()