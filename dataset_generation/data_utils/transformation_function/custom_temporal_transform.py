import math
import argparse
from PIL import Image
import torchvision.transforms as transforms
import random
import torch

def centralized_temporal_subsample(video, max_frames):
    # Size of video is c f h w
    print("Center crop!")
    print(video.shape)
    c, f, h, w = video.shape
    
    if max_frames >= f:
        return video
    
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
    
def stride_temporal_subsample(video, stride=2, max_frames=16):
    # 1. Apply the stride first
    # video shape: [C, F, H, W]
    video = video[:, ::stride, :, :]
    
    # 2. Get the new frame count
    c, f, h, w = video.shape
    
    # CASE 1: Exactly what we need
    if f == max_frames:
        return video
    
    # CASE 2: Post-stride video is too long -> Clip it
    if f > max_frames:
        return video[:, :max_frames, :, :]
    
    # CASE 3: Post-stride video is too short -> Pad or Repeat
    # We use 'repeat' to loop the video until it reaches max_frames
    if f < max_frames:
        last_frame = video[:, -1:, :, :] 
                
        # Calculate how many more frames we need
        diff = max_frames - f
        
        # Repeat that one frame 'diff' times
        padding = last_frame.repeat(1, diff, 1, 1)
        
        # Concatenate original (short) video with the padding
        video = torch.cat((video, padding), dim=1)
        
        return video
    return video

def main():
  print("Custom temporal transformation")

    
if __name__ == "__main__":
    main()