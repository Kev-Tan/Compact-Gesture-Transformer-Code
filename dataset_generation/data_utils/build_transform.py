import torch
from torchvision import transforms
import argparse
from PIL import Image
from torchvision.utils import save_image



def standard_transform(args, is_train):
    
    t = []
    
    
    t.append(transforms.ToTensor())
    
    if(args.random_erasing):
        t.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)))
        
    if(args.random_horizontal_flip):
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        
    if(args.random_cropping):
        t.append(transforms.RandomApply([transforms.RandomResizedCrop(112)], p=0.5))
        
    if(args.restricted_rotation):
        print("Doing restricted rotation")
        t.append(transforms.RandomApply([transforms.RandomRotation(degrees=[-10, 10])], p=0.5))
        
    if(args.shear):
        t.append(transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=15)], p=0.5))
        
    if(args.translate):
        t.append(transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], p=0.5))

    transform = transforms.Compose(t)
    print(transform)
    return transform

def main():
    print("Testing on a single image")
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_erasing', action="store_true", help = "trigger random erasing operations")
    parser.add_argument('--random_horizontal_flip', action="store_true")
    parser.add_argument('--random_cropping', action="store_true")
    parser.add_argument('--restricted_rotation', action="store_true")
    parser.add_argument('--shear', action="store_true")
    parser.add_argument('--translate', action="store_true")


    args = parser.parse_args()
    
    transform = standard_transform(args, True)
    img_addr = r"D:\\School\\Lab\\Compact-Gesture-Transformer-Code\\data\\Briareo\\rgb\\train\\011\\g06\\01\\rgb\\040_rgb.png"

    image = Image.open(img_addr).convert('RGB')
    transformed_image = transform(image)
    transformed_image.show()
    save_image(transformed_image, r'output_image.png')
    


if __name__ == "__main__":
    main()