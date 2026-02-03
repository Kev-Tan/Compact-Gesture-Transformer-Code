import argparse
import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
import math
import numpy as np
import cv2



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



def get_set(args, split, transform=None):
    if(args.dataset_name=='Briareo'):
        print("Retrieve Briareo Dataset")
        ds = Briareo(
                     configer=None,
                     path=args.dataset_root_path, 
                     split=split, data_type="rgb", 
                     transforms = transform, 
                     n_frames = args.num_frames,
                     )
        ds.num_classes = 12
        print(f"{args.dataset_name} {split} split. N={ds.__len__()}, K={ds.num_classes}.")
        return ds

def main():
    
    # python .\build_dataset.py --dataset_root_path "D:\School\Lab\Compact-Gesture-Transformer-Code\Briareo_rgb" --dataset_name Briareo 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required=True,
                        help='path to dataset root')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--save_name_train', type=str, default='train_val')
    parser.add_argument('--save_name_test', type=str, default='test')
    parser.add_argument('--num_frames', type=int, default=30)
    args = parser.parse_args()
    print(args.dataset_name)
    train_loader = get_set(args, 'train')
    sample = next(iter(train_loader))
    print(sample[0])


if __name__ == '__main__':
    main()
