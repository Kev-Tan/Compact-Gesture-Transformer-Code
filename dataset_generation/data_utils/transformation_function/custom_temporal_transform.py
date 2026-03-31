import math
import argparse
from PIL import Image
import torchvision.transforms as transforms
import random

def centralized_temporal_subsample(video, max_frames):
    # Size of video is c f h w
    print("Center crop!")
    print(video.shape)
    c, f, h, w = video.shape
    
    center_of_list = math.floor(f/2)
    crop_limit = math.floor(max_frames / 2)
    start = center_of_list - crop_limit
    end = center_of_list + crop_limit
    
    video = video[:,start:end+1,:,:]
    print("New video shape")
    print(video.shape)
    print(f"Start index {start}, End index {end}")
    return video

def random_temporal_subsample(video, max_frames):
        c, f, h, w = video.shape
        if max_frames >= f:
            return video

        start = random.randint(0, f - max_frames)
        end = start + max_frames
        return video[:, start:end, :, :]
    
def stride_temporal_subsample(video, stride = 2, max_frames = None):
    video = video[:, ::stride, :, :]
    if max_frames is not None:
        video = video[:, :max_frames, :, :]
    return video

def main():
  print("Custom temporal transformation")

    
if __name__ == "__main__":
    main()