import torch
import torch.nn as nn
import kornia.augmentation as K
import argparse



class KorniaVideoTransform(nn.Module):
    def __init__(
        self,
        args,
        jitter_p: float = 0.5,
        rotation_p: float = 0.3,
        blur_p: float = 0.2,
        motion_blur_p: float = 0.3,
    ):
        super().__init__()
        
        transform = []
        transform.append(K.Resize((args.image_size, args.image_size)))
        
        if args.random_affine:
            transform.append(
                K.RandomAffine(
                    degrees=3.0,
                    translate=(0.02, 0.02),
                    scale=(0.98, 1.02),
                    shear=2.0,
                    p=0.3,
                )
            )

        if args.color_jitter:
            transform.append(K.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.2,
                p=jitter_p,
            ))
        if args.random_rotation:
            transform.append(K.RandomRotation(
                degrees=5.0,
                p=rotation_p,
            ))
        if args.random_gaussian_blur:
            transform.append(
            K.RandomGaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.1, 0.5),
                p=blur_p,
            )
        )
        if args.random_motion_blur:
            transform.append(K.RandomMotionBlur(
            kernel_size=(3, 15),   # length of blur streak
            angle=(-10., 10.),     # random direction
            direction=(-1., 1.),   # random forward/backward
            p=0.6)
            )
            
        if args.random_gaussian_noise:
            transform.append(
                K.RandomGaussianNoise(
                    mean=0.0,
                    std=0.2,
                    p=0.2,
                )
            )
        
        if args.random_resized_cropping:
            transform.append(
            K.RandomResizedCrop(
                size=(224, 224),
                scale=(0.5, 0.9),
                ratio=(0.75, 1.33),
                p=0.5
            )
        )
            
        if args.elastic_transformation:
            transform.append(
            K.RandomElasticTransform(kernel_size=(63, 63), 
                                                        sigma=(32.0, 32.0), 
                                                        alpha=(1.0, 1.0), 
                                                        align_corners=False, 
                                                        # resample=Resample.BILINEAR.name,
                                                        padding_mode='zeros', 
                                                        same_on_batch=False,
                                                        p=0.5,
                                                        keepdim=False)
            )
                        
        if args.random_perspective:
            transform.append(
               K.RandomPerspective(distortion_scale=0.5,
                                   same_on_batch=False, 
                                   align_corners=False,
                                   p=0.5, keepdim=False, 
                                   sampling_method='basic')
            )
            
        if args.random_erasing:
            transform.append(
                K.RandomErasing(scale=(0.02, 0.33),
                                ratio=(0.3, 3.3), 
                                value=0.0, 
                                same_on_batch=False,
                                p=0.5, keepdim=False)
            )
            

        self.transform = K.VideoSequential(
            *transform   ,
            same_on_frame=True,
            data_format="BCTHW",
        )
        

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip: (C, T, H, W), usually uint8 in [0, 255]
        returns: (C, T, H, W), float32 in [0, 1]
        """
        
        # print("Shape of clip is ", clip.shape)
        
        if clip.ndim != 4:
            raise ValueError(f"Expected clip shape (C, T, H, W), got {clip.shape}")

        # Convert uint8 -> float32 and normalize to [0, 1]
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
        else:
            clip = clip.float()

        # Kornia VideoSequential expects 5D input, more specifically (B, T, C, H, W)
        clip = clip.unsqueeze(0)   # (1, C, T, H, W)
        clip = self.transform(clip)
        clip = clip.squeeze(0)     # (C, T, H, W)

        return clip
    

def main():
    print("Lmao")
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_jitter', action="store_true")
    parser.add_argument('--random_rotation', action="store_true")
    parser.add_argument('--random_gaussian_blur', action="store_true")
    parser.add_argument('--random_motion_blur', action="store_true")
    args = parser.parse_args()
    transform = KorniaVideoTransform(args)
    print(transform)
    
    
    

if __name__ == "__main__":
    main()