
from torchvision.transforms import Compose, Lambda, RandomCrop
from torchvision.transforms._transforms_video import NormalizeVideo
from pytorchvideo.transforms import UniformTemporalSubsample



def standard_transform(args):
    
    t = []
    
    if(args.uniform_temporal_subsample):
        t.append(UniformTemporalSubsample(args.uniform_temporal_subsample))
        
    if(args.random_crop):
        t.append(RandomCrop(args.random_crop))

    t.append(Lambda(lambda x: x / 255.0))
    
    transform = Compose(t)
    return transform