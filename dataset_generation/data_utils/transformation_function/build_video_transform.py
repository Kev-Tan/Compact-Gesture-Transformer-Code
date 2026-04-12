from torchvision.transforms import Compose
from torchvision import transforms
from .kornia_transform import KorniaVideoTransform
from pytorchvideo.transforms import UniformTemporalSubsample
import argparse
from torchvision import transforms


def standard_transform(args):
    
    transform_list = []
    print(args)
    
    if args.uniform_temporal_subsample:
        transform_list.append(UniformTemporalSubsample(args.n_frames))
        
    
    transform_list.append(KorniaVideoTransform(args))
    train_transform = Compose(transform_list)
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_transform = Compose([
    UniformTemporalSubsample(args.n_frames),
    transforms.Resize(256),
    transforms.CenterCrop(args.image_size),
    # transforms.Normalize(mean, std),
])
    
    return train_transform, test_transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uniform_temporal_subsample', type=int, default=None, metavar='N',help='Determine number of frames subsampled' )
    parser.add_argument('--color_jitter', action="store_true")
    parser.add_argument('--random_rotation', action="store_true")
    parser.add_argument('--random_gaussian_blur', action="store_true")
    parser.add_argument('--random_motion_blur', action="store_true")
    args = parser.parse_args()
    transform = standard_transform(args)
    print(transform)
    
    
    

if __name__ == "__main__":
    main()