import argparse
import ast
import copy
import functools
import glob
import json
import math
import os
import pdb
import random
from pathlib import Path

import cv2
import einops
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from numpy.random import randint
from PIL import Image
from pytorchvideo.transforms import UniformTemporalSubsample
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import get_image_backend
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo
from torchvision.utils import save_image


# Relative import
from .generate_csv import generate_csv
from .build_video_transform import standard_transform
from .temporal_transform import sampling
from .custom_temporal_transform import centralized_temporal_subsample, random_temporal_subsample, stride_temporal_subsample

class DatasetVideoTarget(data.Dataset):
    def __init__(self, args, root, split, transform = None):  
        self.train_transform = transform[0]
        self.test_transform = transform[1]
        self.args = args
        self.root = root
        self.split = split
        self.images_folder = root
        self.n_frames = args.n_frames
        print("---self.root", self.root)
        path_csv = os.path.join(self.root, f'{split}.csv')
        self.df = pd.read_csv(path_csv, sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()
        self.num_classes = len(set(self.targets))
        
        
        fixed_data = []
        for i, record in enumerate(self.data):
            record = ast.literal_eval(record)
            fixed_data.append(record)
        
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
            path_frame = os.path.join(self.root,  p[0])
            torch.manual_seed(seed)

            # H, W, C
            img = Image.open(path_frame)
            rgb_img = img.convert("RGB")
            clip.append(rgb_img)
        
        clip = np.array(clip)
        clip = einops.rearrange(clip,"f h w c -> c f h w")
        clip = torch.from_numpy(clip)
        clip = clip.float() / 255.0
        
        
        if(self.args.centralized_temporal_subsample):
            print(paths[0])
            clip = centralized_temporal_subsample(clip, self.n_frames)
        if(self.args.random_temporal_subsample):
            print(paths[0])
            clip = random_temporal_subsample(clip, self.n_frames)
        if(self.args.stride_temporal_subsample):
            print(paths[0])
            clip = stride_temporal_subsample(clip, 2,  self.n_frames)
        
        
        if(self.train_transform and (self.split=="train" or self.split=="val")):
            clip = self.train_transform(clip)
        else:
            clip = self.test_transform(clip)
        
        clip = einops.rearrange(clip,"f c h w -> c f h w")
        label = torch.LongTensor(np.asarray([label]))
        label = label.squeeze(-1)
        return clip.float(), label

def vis_dataset(args):

    transform = standard_transform(args)
    result_dir = r"dataset_generation\\tester_images"
    train_set = DataLoader(DatasetVideoTarget(args = args, root=args.dataset_root_path, split='train', transform=transform), batch_size=1)
    val_set = DataLoader(DatasetVideoTarget(args = args, root=args.dataset_root_path, split='val', transform=transform), batch_size=1)
    test_set = DataLoader(DatasetVideoTarget(args = args, root=args.dataset_root_path, split='test', transform=transform), batch_size=1)
    for split, loader in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
        for idx, (images, _) in enumerate(loader):
            batch_size, frames, channels, height, width = images.shape
            images = einops.rearrange(images,"b f c h w -> (b f) c h w")
            
            fp = os.path.join(result_dir, split, f'{idx}.png')
            save_image(images, fp, nrow=int(math.sqrt(images.shape[0])))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required = True, help="path to dataset root")
    parser.add_argument('--dataset_name', type=str, required = True, help = "name of the dataset")
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--uniform_temporal_subsample',type=int, default=None, metavar='N',help='Apply uniform temporal subsampling with N frames' )
    parser.add_argument('--n_frames',type=int, default=None, metavar='N',help='number of frames per batch' )
    parser.add_argument('--centralized_temporal_subsample', action="store_true")  
    parser.add_argument('--random_temporal_subsample', action="store_true")  
    parser.add_argument('--stride_temporal_subsample', action="store_true")  
    parser.add_argument('--color_jitter', action="store_true")
    parser.add_argument('--random_rotation', action="store_true")
    parser.add_argument('--random_gaussian_blur', action="store_true")
    parser.add_argument('--random_motion_blur', action="store_true")
    parser.add_argument('--random_affine', action="store_true")
    parser.add_argument('--random_gaussian_noise', action="store_true")
    parser.add_argument('--random_resized_cropping', action="store_true")
    parser.add_argument('--elastic_transformation', action="store_true")
    parser.add_argument('--random_perspective', action="store_true")
    parser.add_argument('--random_erasing', action="store_true")
    parser.add_argument('--image_size',type=int, default=None, metavar='N',help='Convert the image to a defined image sized' )

    
    args = parser.parse_args()
    

    file_path = Path(os.path.join(args.dataset_root_path, "train.csv"))
    if not os.path.exists(file_path):
        print("Generate train")
        generate_csv(args, split='train')
        
    file_path = Path(os.path.join(args.dataset_root_path, "test.csv"))
    if not os.path.exists(file_path):
        print("Generate test")
        generate_csv(args, split='test') 
        
    file_path = Path(os.path.join(args.dataset_root_path, "val.csv"))
    if not os.path.exists(file_path):
        print("Generate val")
        generate_csv(args, split='val')
        
    if(args.visualize):
        vis_dataset(args)
    else:
        print("Not visualized")
            
    
    


if __name__ == '__main__':
    main()