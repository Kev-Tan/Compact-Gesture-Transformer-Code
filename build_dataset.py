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

def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Flow':
                return img.convert('L')
            elif modality == 'Depth':
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality, sample_duration, image_loader):
    video = []
    if modality == 'RGB':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'Depth':

        for i in frame_indices:
            image_path = os.path.join(video_dir_path.rsplit(os.sep,2)[0] , 'Depth','depth' + video_dir_path[-1], '{:06d}.jpg'.format(i) )
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'RGB-D':
        for i in frame_indices: # index 35 is used to change img to flow
            image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
            image_path_depth = os.path.join(video_dir_path.rsplit(os.sep,2)[0] , 'Depth','depth' + video_dir_path[-1], '{:06d}.jpg'.format(i) )
    
            image = image_loader(image_path, 'RGB')
            image_depth = image_loader(image_path_depth, 'Depth')
            if os.path.exists(image_path):
                video.append(image)
                video.append(image_depth)
            else:
                print(image_path, "------- Does not exist")
                return video
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset in subset:
            label = value['annotations']['label']
            video_names.append(key.split('_')[0])
            annotations.append(value['annotations'])
            
    # print(video_names, annotations)

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    if type(subset)==list:
        subset = subset
    else:
        subset =  [subset]
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    list_subset = ''
    for x in subset:
        list_subset += x+',' 
    print("[INFO]: EgoGesture Dataset - " + list_subset + " is loading...")
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        
        if not os.path.exists(video_path):
            print(video_path + " does not exist")
            continue

        #### Add more frames from start end end
        begin_t = int(float(annotations[i]['start_frame']))
        end_t = int(float(annotations[i]['end_frame']))
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)
    return dataset, idx_to_class


class EgoGesture(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 get_loader=get_default_video_loader):

        if subset == 'training':
            subset = ['training', 'validation']
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)
        oversample_clip =[]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        
        
        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)

        # PIL -> Tensor
        clip = [
            torch.from_numpy(np.array(img)).permute(2, 0, 1)
            for img in clip
        ]

                
            
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return clip, target

    def __len__(self):
        return len(self.data)
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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Flow':
                return img.convert('L')
            elif modality == 'Depth':
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality, sample_duration, image_loader):
    video = []
    if modality == 'RGB':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'Depth':

        for i in frame_indices:
            image_path = os.path.join(video_dir_path.rsplit(os.sep,2)[0] , 'Depth','depth' + video_dir_path[-1], '{:06d}.jpg'.format(i) )
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'RGB-D':
        for i in frame_indices: # index 35 is used to change img to flow
            image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
            image_path_depth = os.path.join(video_dir_path.rsplit(os.sep,2)[0] , 'Depth','depth' + video_dir_path[-1], '{:06d}.jpg'.format(i) )
    
            image = image_loader(image_path, 'RGB')
            image_depth = image_loader(image_path_depth, 'Depth')
            if os.path.exists(image_path):
                video.append(image)
                video.append(image_depth)
            else:
                print(image_path, "------- Does not exist")
                return video
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset in subset:
            label = value['annotations']['label']
            video_names.append(key.split('_')[0])
            annotations.append(value['annotations'])
            
    # print(video_names, annotations)

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    if type(subset)==list:
        subset = subset
    else:
        subset =  [subset]
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    list_subset = ''
    for x in subset:
        list_subset += x+',' 
    print("[INFO]: EgoGesture Dataset - " + list_subset + " is loading...")
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        
        if not os.path.exists(video_path):
            print(video_path + " does not exist")
            continue

        #### Add more frames from start end end
        begin_t = int(float(annotations[i]['start_frame']))
        end_t = int(float(annotations[i]['end_frame']))
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)
    return dataset, idx_to_class


class EgoGesture(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 get_loader=get_default_video_loader):

        if subset == 'training':
            subset = ['training', 'validation']
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)
        oversample_clip =[]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        
        
        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)

        # PIL -> Tensor
        clip = [
            torch.from_numpy(np.array(img)).permute(2, 0, 1)
            for img in clip
        ]

                
            
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return clip, target

    def __len__(self):
        return len(self.data)



class Briareo(Dataset):
    """Briareo Dataset class"""
    def __init__(self, path, split="train", data_type='depth', transforms=None, n_frames=30, optical_flow=False):
        """Constructor method for Briareo Dataset class

        Args:
            # configer (Configer): Configer object for current procedure phase (train, test, val)
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
            
            print(center_of_list, crop_limit, start,end)
            
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
    
class CDEFG(Briareo):
    pass

def get_set(args, split, transform=None):
    if(args.dataset_name=='briareo'):
        # train, test, val
        print("Retrieve Briareo Dataset")
        ds = Briareo(
                     path=args.dataset_root_path, 
                     split=split, data_type="rgb", 
                     transforms = transform, 
                     n_frames = args.num_frames,
                     )
        ds.num_classes = 12
        print(f"{args.dataset_name} {split} split. N={ds.__len__()}, K={ds.num_classes}.")
        return ds
    elif(args.dataset_name=='cdefg'):
        # train, test, val
        print("Retrieve CDEFG Dataset")
        ds = CDEFG(
                path=args.dataset_root_path, 
                split=split, data_type="rgb", 
                transforms = transform, 
                n_frames = args.num_frames,
                )
        ds.num_classes = 105
        print(f"{args.dataset_name} {split} split. N={ds.__len__()}, K={ds.num_classes}.")
        return ds
    elif(args.dataset_name=='egogesture'):
        # training, testing, validation
        print("Retrieve Egogesture dataset")
        egoGestureSplit = None
        if(split=="train"):
            egoGestureSplit = "training"
        ds = EgoGesture(root_path=args.dataset_root_path, annotation_path=args.annotation_path,subset=egoGestureSplit,sample_duration=args.num_frames)
        return ds
    
    
def vis_dataset_single_image(loader):
    
    dataloader = DataLoader(loader)


    for idx, (samples, _) in enumerate(dataloader):
        # print("len of samples", len(samples))
        print("--BATCH: ", idx)
        # Iterate through the gestures inside a batch
        for idx, images in enumerate(samples): 
            # 93 frames in a gesture 
            print("--Image:", idx)
            print(len(images))
            for idx, frame in enumerate(images):
                print("--Frame;", idx)
                print(frame.size())
                image = np.squeeze(frame)
                plt.imshow(image)
                plt.show()
                print(type(image))
        return
    
# torch vision utils save image

def main():
    
    # python .\build_dataset.py --dataset_root_path "D:\School\Lab\Compact-Gesture-Transformer-Code\Briareo_rgb" --dataset_name Briareo 
    # Annotation path for egogesture: "/home/mislab/Charlene/annotation_EgoGesture/trainlistall.txt"
    # Dataset path for egogesture: "/home/mislab/Charlene/frames/"
    # JSON Egogesture: "/home/mislab/Charlene/annotation_EgoGesture/egogestureall.json"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required=True,
                        help='path to dataset root')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--save_name_train', type=str, default='train_val')
    parser.add_argument('--save_name_test', type=str, default='test')
    parser.add_argument('--num_frames', type=int, default=40)
    parser.add_argument('--annotation_path', type=str, default="/home/mislab/Charlene/annotation_EgoGesture/egogestureall.json")
    args = parser.parse_args()
    print(args.dataset_name)
    train_loader = get_set(args, 'train')
    vis_dataset_single_image(train_loader)
    sample = next(iter(train_loader))
    print(sample[0])


if __name__ == '__main__':
    main()
