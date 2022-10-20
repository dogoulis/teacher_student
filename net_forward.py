import argparse
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


import timm
import torch
import torch.nn as nn

from dataset import pytorch_dataset, augmentations
from torch.utils.data.dataloader import DataLoader

# parser:
parser = argparse.ArgumentParser()



parser.add_argument(
    "--model",
    type=str,
    required=True,
    help='Model used for the experiment.',
    choices=['resnet50', 'swin-tiny', 'vit-small']
)

parser.add_argument(
    "--device",
    type=str,
    default='cpu',
    metavar="device",
    help="Device used for the inference."
)

parser.add_argument(
    "--img_dir",
    type=str,
    metavar='img_dir',
    required=True,
    help='Directory for input image.'
)

parser.add_argument(
    "--weights",
    type=str,
    metavar='weights',
    required=True,
    help='The directory for the weights of the model.'
)

args = parser.parse_args()

def get_validation_augmentations(resize_size=256, crop_size=224):
    return A.Compose(
        [
            A.augmentations.geometric.resize.Resize(resize_size, resize_size),
            A.augmentations.crops.transforms.CenterCrop(crop_size, crop_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


@torch.no_grad()
def testing(model, path, transforms, device):
    
    model.eval().to(device)


    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tr_img = transforms(image=img)
    image = tr_img["image"]

    
    image = torch.unsqueeze(image, 0).to(device)
    output = model(image)
    output = torch.sigmoid(output)
    print(output)
    return output



# main def:
def main():


    # initialize model:
    if args.model == 'resnet50':
        model = timm.create_model('resnet50', num_classes=1, pretrained=True)
    elif args.model == 'swin-tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=1, pretrained=True)
    elif args.model == 'vit-small':
        model = timm.create_model('vit_small_patch16_224', num_classes=1, pretrained=True)
    else:
        print('No selected model')
    # load weights:
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    # model = model.eval().to(args.device)

    # defining transforms:
    transforms = get_validation_augmentations()

    output = testing(
        model=model, transforms=transforms, path=args.img_dir, device=args.device)



    print(f'Finished Testing with output = {output}')


if __name__ == '__main__':
    main()