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
import einops
from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo
from pytorchvideo.transforms import UniformTemporalSubsample




# Relative import
from .generate_csv import generate_csv
from .pytorch_video_transform import standard_transform
from .temporal_transform import sampling

class DatasetVideoTarget(data.Dataset):
    def __init__(self, root, split, transform = None , n_frames = 20):
        # Split is train, test, val
        
        self.transform = transform
        self.root = root
        self.split = split
        self.images_folder = root
        self.n_frames = n_frames
        print("---self.root", self.root)
        path_csv = os.path.join(self.root, f'{split}.csv')
        self.df = pd.read_csv(path_csv, sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()
        self.num_classes = len(set(self.targets))
        # self.num_classes = max(self.targets) + 1
        # print("NUM CLASSES IS", self.num_classes)
        # print(len(set(self.targets)))
        
        
        # self.data = sampling(self.data, 30)
        
        # Stores the path to images selected after sampling
        # fixed_data = sampling(self.data, n_frames)
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
        
        # Passing it to transformation function makes it (c, h, w) by default)
        # If we did not pass it through the transformation function, we have to transpose it since it's (h, w, c) if I'm not mistaken
        clip = np.array(clip)
        # print("Shape of clip before is", clip.shape)
        # clip = einops.rearrange(clip,"f h w c -> f c h w")
        clip = einops.rearrange(clip,"f h w c -> c f h w")
        clip = torch.from_numpy(clip)
        # print("Shape of clip before is before transform", clip.shape)
        clip = self.transform(clip)
        clip = einops.rearrange(clip,"f c h w -> c f h w")
        # print("Shape of clip is ", clip.shape)

        label = torch.LongTensor(np.asarray([label]))
        label = label.squeeze(-1)
        return clip.float(), label

def vis_dataset(args):

    # print("Testing", transform_function)
    transform = standard_transform(args)
    result_dir = r"dataset_generation\\tester_images"
    train_set = DataLoader(DatasetVideoTarget(root=args.dataset_root_path, split='train', transform=transform), batch_size=1)
    test_set = DataLoader(DatasetVideoTarget(root=args.dataset_root_path, split='test', transform=transform), batch_size=1)
    val_set = DataLoader(DatasetVideoTarget(root=args.dataset_root_path, split='val'), batch_size=1)
    for split, loader in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
        for idx, (images, _) in enumerate(loader):
            # print("Shape of images is ", images.shape)
            # (B, F, C, H, W)
            batch_size, frames, channels, height, width = images.shape
            # images = einops.rearrange(images, "b c f h w -> b c t h w")
            # (batch, frames, channel, height, width)
            # images = images.permute(0, 1, 4, 2, 3)
            
            # Convention is B,F,C,H,W
            # images = images.view(-1, channels, height, width)
            images = einops.rearrange(images,"b f c h w -> (b f) c h w")
            
            # images = images.view(batch_size * frame, channels, height, width )
            # print("Revised image shape is ", images.shape)
            
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
    # parser.add_argument('--random_erasing', action="store_true", help = "trigger random erasing operations")
    # parser.add_argument('--random_horizontal_flip', action="store_true")
    # parser.add_argument('--random_cropping', action="store_true")
    # parser.add_argument('--restricted_rotation', action="store_true")
    # parser.add_argument('--shear', action="store_true")
    # parser.add_argument('--translate', action="store_true")
    parser.add_argument('--uniform_temporal_subsample',type=int, default=None, metavar='N',help='Apply uniform temporal subsampling with N frames' )
    # parser.add_argument('--normalize', type= bool, default=True, help="Determine is normalization will be applied")
    # parser.add_argument('--random_crop', type=int, default=None, metavar='N',help='Apply random crop to a given size' )
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