import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch.utils.data as data
import math
import ast
import cv2
import matplotlib as plt
from PIL import Image
import argparse
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset
import torch
class Briareo_csv(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform = transform
        new_images_path = None
        if split=="val":
            new_images_path = os.path.join(root, 'rgb', 'splits', 'train', 'rgb_val.npz')
        else:
            new_images_path = os.path.join(root, 'rgb', 'splits', split, f'rgb_{split}.npz')
        self.image_path = np.load(new_images_path, allow_pickle=True)['arr_0']
        self.image_path.tolist()
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.image_path[idx]['label'], self.image_path[idx]['data']
    
def generate_csv(root, split):
    dataset_obj = Briareo_csv(root = root, split = split)
    dic_target_img_dir = {}
    for index, (id, path) in enumerate(DataLoader(dataset_obj)):
        dic_target_img_dir[index] = {'class_id': id.item(), 'dir': path}
        
    df = pd.DataFrame.from_dict(dic_target_img_dir, orient='index')
    fp = os.path.join(root, f"{split}.csv")
    df.to_csv(fp, header = True, index= False)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required = True, help="path to dataset root")
    parser.add_argument('--dataset_name', type=str, required = True, help = "name of the dataset")
    args = parser.parse_args()
    
    if(args.dataset_name =='briareo'):
        dataset_train = Briareo_csv(root = args.dataset_root_path, split='train')
        dataset_val = Briareo_csv(root = args.dataset_root_path, split='val')
        dataset_test = Briareo_csv(root = args.dataset_root_path, split='test')
        
        generate_csv(args.dataset_root_path, split='train')
        generate_csv(args.dataset_root_path, split='val')
        generate_csv(args.dataset_root_path, split='test')

if __name__ == "__main__":
    main()