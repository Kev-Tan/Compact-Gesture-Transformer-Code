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
from pathlib import Path


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
    
    
class Egogesture_csv(Dataset):
        def __init__(self, root, split, transform = None):
            self.transform = None
            self.image_path = []
            self.labels = []
            if split == 'train':
                images_info_path = os.path.join(root, 'trainlistbinary.txt')
                with open(images_info_path, 'r') as f:
                    images_info_array = f.readlines()
            elif split == "test":
                images_info_path = os.path.join(root, 'testlistbinary.txt')
                with open(images_info_path, 'r') as f:
                    images_info_array = f.readlines()
            elif split == "val":
                images_info_path = os.path.join(root, 'vallistbinary.txt')
                with open(images_info_path, 'r') as f:
                    images_info_array = f.readlines()
                
            # In case it takes a long time, try taking the first n elements of images_info_array
            # This code is very inefficient but it's a byproduct of the label and path structure
            count = 0
            
            for info in images_info_array:
                print(count)
                count+=1
                try:
                    # print(info)
                    split_info = info.split()
                    path_to_images = split_info[0]
                    is_gesture = int(split_info[1])
                    starting_frame = int(float(split_info[2]))
                    ending_frame = int(float(split_info[3]))
                    
                    subject = path_to_images.split('/')[0]
                    subject = subject.lower()
                    
                    scene = path_to_images.split('/')[1]
                    rgb_group = path_to_images.split('/')[3]
                    
                    
                    if(is_gesture == 2):
                        csv_files_path = os.path.join(root, 'labels', subject, scene)
                        labels_information = []
                        for file in os.listdir(csv_files_path):
                            if file.endswith(".csv"):
                                correct_file_path = os.path.join(csv_files_path, file)
                                csv_content = pd.read_csv(correct_file_path, header=None)
                                for _, row in csv_content.iterrows():
                                    labels_information.append(row.tolist())
                                
                                gestures_path_list = []
                                for columns in labels_information:
                                    # print(columns)
                                    # print(starting_frame)
                                    # print(ending_frame)
                                    if columns[1]==starting_frame and columns[2]==ending_frame:
                                        for i in range(starting_frame, ending_frame + 1):
                                            path = os.path.join('frames', subject, scene, "Color", rgb_group, f"{i:06}.jpg")
                                            gestures_path_list.append(path)
                                        self.labels.append(columns[0])
                                        break
                                
                                if(gestures_path_list):
                                    self.image_path.append(gestures_path_list)
                                    break
                                
                except:
                    print("ERROR ON PATH", path_to_images)
                    
                
            # return
        
        def __len__(self):
            return len(self.image_path)
        
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            return self.labels[idx], self.image_path[idx]
            
        def check_paths_and_labels(self):
            print(self.image_path)     
            print(self.labels)

    
def generate_csv(args, split):
    if(args.dataset_name == 'briareo'):
        dataset_obj = Briareo_csv(root = args.dataset_root_path, split = split)
    if(args.dataset_name == 'egogesture'):
        dataset_obj = Egogesture_csv(root = args.dataset_root_path, split = split)
    dic_target_img_dir = {}
    for index, (id, path) in enumerate(DataLoader(dataset_obj)):
        try:
            dic_target_img_dir[index] = {'class_id': id.item(), 'dir': path}
        except:
            class_id = int(id[0])
            dic_target_img_dir[index] = {'class_id': class_id, 'dir': path}
            # print("ERRROR")
            # print(id)
            # print(path)
        
    df = pd.DataFrame.from_dict(dic_target_img_dir, orient='index')
    fp = os.path.join(args.dataset_root_path, f"{split}.csv")
    df.to_csv(fp, header = True, index= False)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required = True, help="path to dataset root")
    parser.add_argument('--dataset_name', type=str, required = True, help = "name of the dataset")
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

if __name__ == "__main__":
    main()