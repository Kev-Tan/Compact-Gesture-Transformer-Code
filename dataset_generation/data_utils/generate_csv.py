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
    
    
class Egogesture_csv(Dataset):
        def __init__(self, root, split, transform = None):
            self.transform = None
            self.image_path = []
            self.labels = []
            if split == 'train':
                # Loads txt file that contains information the path and which frame is a gesture
                images_info_path = os.path.join(root, 'trainlistbinary.txt')
                with open(images_info_path, 'r') as f:
                    images_info_array = f.readlines()
            elif split == "test":
                # Loads txt file that contains information the path and which frame is a gesture
                images_info_path = os.path.join(root, 'testlistbinary.txt')
                with open(images_info_path, 'r') as f:
                    images_info_array = f.readlines()
            elif split == "val":
                # Loads txt file that contains information the path and which frame is a gesture
                images_info_path = os.path.join(root, 'vallistbinary.txt')
                with open(images_info_path, 'r') as f:
                    images_info_array = f.readlines()
                
            print(len(images_info_array))
            # Iterate and separate it based on the space
            for info in images_info_array[:10]:
                try:
                    split_info = info.split()
                    path_to_images = split_info[0]
                    is_gesture = int(split_info[1])
                    starting_frame = int(float(split_info[2]))
                    ending_frame = int(float(split_info[3]))
                    
                    subject = path_to_images.split('/')[0]
                    subject = subject.lower()
                    
                    scene = path_to_images.split('/')[1]
                    rgb_group = path_to_images.split('/')[3]
                    
                    
                    # print("Path is", path_to_images)
                    # print("Subject is:", subject)
                    # print("Scene is: ", scene)
                    # print("Is gesture: ", True if is_gesture == 2 else False)
                    # print("Starting frame: ", starting_frame)
                    # print("Ending frame: ", ending_frame)
                    
                    
                    
                    # If it's a gesture, then open all the csv files within the path and scene
                    if(is_gesture == 2):
                        csv_files_path = os.path.join(root, 'labels', subject, scene)
                        labels_information = []
                        for file in os.listdir(csv_files_path):
                            if file.endswith(".csv"):
                                correct_file_path = os.path.join(csv_files_path, file)
                                csv_content = pd.read_csv(correct_file_path, header=None)
                                # Store the csv files within an array called labels_information
                                for _, row in csv_content.iterrows():
                                    labels_information.append(row.tolist())
                                
                                gestures_path_list = []
                                for columns in labels_information:
                                    print(columns)
                                    print(starting_frame)
                                    print(ending_frame)
                                    if columns[1]==starting_frame and columns[2]==ending_frame:
                                        for i in range(starting_frame, ending_frame + 1):
                                            path = os.path.join('frames', subject, scene, "Color", rgb_group, f"{i:06}.jpg")
                                            gestures_path_list.append(path)
                                        self.labels.append(columns[0])
                                        break
                                
                                self.image_path.append(gestures_path_list)
                                if(gestures_path_list):
                                    break
                                
                                
                                # print(labels_information)
                    
                    
                    
                except:
                    print("ERROR ON PATH", path_to_images)
                    
                
            return
        
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
        dic_target_img_dir[index] = {'class_id': id.item(), 'dir': path}
        
    df = pd.DataFrame.from_dict(dic_target_img_dir, orient='index')
    fp = os.path.join(args.dataset_root_path, f"{split}.csv")
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
        
    elif(args.dataset_name == 'egogesture'):
        dataset_train = Egogesture_csv(root = args.dataset_root_path, split='train')
        dataset_val = Egogesture_csv(root = args.dataset_root_path, split='val')
        dataset_test = Egogesture_csv(root = args.dataset_root_path, split='test')
        
        generate_csv(args, split='train')
        generate_csv(args, split='test')
        generate_csv(args, split='val')
        dataset_train.check_paths_and_labels()

if __name__ == "__main__":
    main()