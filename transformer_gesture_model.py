# Import everything that is necessary
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
from torch.utils.data.dataset import Dataset
import math
import torch
from pathlib import Path
import cv2
import numpy as np
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from urllib.request import urlopen
from PIL import Image
import timm
from timm import create_model
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt

# Decide if we use timm_backbone or not
timm_backbone = True
model_name = "resnet18.a1_in1k"

hyperparams = {
    # Network params
    'n_classes': 12,
    'pretrained': True,
    'n_head': 8,
    'dropout_backbone': 0.1,  # dropout2d
    'dropout_transformer': 0.5,  # dropout1d
    'dff': 1024,  # ff_size
    'n_module': 6,
    
    # Solver params (required by ModuleUtilizer)
    'solver': {
        'type': 'AdamW',  # or 'Adam', 'SGD', 'RMSProp'
        'base_lr': 0.0001,
        'weight_decay': 0.0001,
        'momentum': 0.9,  # only used for SGD
        'lr_policy': 'fixed',  # or 'step', 'multistep', 'exp', 'inv'
        'gamma': 0.1,
        'stepvalue': [50, 75]  # for multistep policy
    },
    
    # Checkpoint params (required by ModuleUtilizer)
    'checkpoints': {
        'save_policy': 'best',  # or 'all', 'early_stop'
        'save_name': 'gesture_model',
        'save_dir': './checkpoints/',
        'early_stop': 10  # patience for early stopping
    },
    
    # Other required params
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dataset': 'Briareo',
    'gpu': [0],  # GPU IDs for DataParallel
    'resume': None  # Path to checkpoint to resume from, or None
}

class Briareo(Dataset):
    """Briareo Dataset class"""
    def __init__(self, configer, path, split="train", data_type='depth', transforms=None, n_frames=30, optical_flow=False):
        """Constructor method for Briareo Dataset class

        Args:
            configer (Configer): Configer object for current procedure phase (train, test, val)
            split (str, optional): Current procedure phase (train, test, val)
            data_type (str, optional): Input data type (depth, rgb, normals, ir)
            transform (Object, optional): Data augmentation transformation for every data
            n_frames (int, optional): Number of frames selected for every input clip
            optical_flow (bool, optional): Flag to choose if calculate optical flow or not

        """
        super().__init__()
        self.dataset_path = Path(path)
        self.split = split
        self.data_type = data_type
        self.optical_flow = optical_flow

        self.transforms = transforms
        self.n_frames = n_frames + 1

        print("Loading Briareo {} dataset...".format(split.upper()), end=" ")
        data = np.load(self.dataset_path / "splits" / (self.split if self.split != "val" else "train") /
                                    "{}_{}.npz".format(data_type, self.split), allow_pickle=True)['arr_0']

        # Prepare clip for the selected number of frames n_frame
        fixed_data = list()
        for i, record in enumerate(data):
            paths = record['data']

            center_of_list = math.floor(len(paths) / 2)
            crop_limit = math.floor(self.n_frames / 2)

            start = center_of_list - crop_limit
            end = center_of_list + crop_limit
            paths_cropped = paths[start: end + 1 if self.n_frames % 2 == 1 else end]
            if self.data_type == 'leapmotion':
                valid = np.array(record['valid'][start: end + 1 if self.n_frames % 2 == 1 else end])
                if valid.sum() == len(valid):
                    data[i]['data'] = paths_cropped
                    fixed_data.append(data[i])
            else:
                data[i]['data'] = paths_cropped
                fixed_data.append(data[i])
            # print("Length of cropped data", len(data[i]['data']))
        self.data = np.array(fixed_data)
        print("done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]['data']
        label = self.data[idx]['label']

        clip = list()
        for p in paths:
            img = cv2.imread(str(self.dataset_path / p), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            if self.data_type != "rgb":
                img = np.expand_dims(img, axis=2)
            clip.append(img)

        clip = np.array(clip).transpose(1, 2, 3, 0)


        if self.transforms is not None:
            aug_det = self.transforms.to_deterministic()
            clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)

        clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
        label = torch.LongTensor(np.asarray([label]))
        return clip.float(), label


def build_dataloaders():
    dataset_path = "D:\School\Lab\Compact-Gesture-Transformer-Code\Briareo_rgb"
    train_loader = DataLoader(Briareo(configer=None, path=dataset_path, data_type="rgb", split="train"), batch_size=8,  shuffle=True, num_workers=4)
    test_loader = DataLoader(Briareo(configer=None, path=dataset_path, data_type="rgb", split="test"), batch_size=8,  shuffle=False, num_workers=0)
    val_loader = DataLoader(Briareo(configer=None, path=dataset_path, data_type="rgb", split="val"), batch_size=1,  shuffle=False, num_workers=0)
    return train_loader, test_loader, val_loader

def main():
    import torch;
    print(torch.__version__)
    print(torch.cuda.is_available())
    train_loader, val_loader, test_loader = build_dataloaders()
    for i, batch in enumerate(train_loader):
        samples, target = batch
        Samples = samples.to('cuda')
        print(i, target)
    
    
    
if __name__ == '__main__':
    main()

