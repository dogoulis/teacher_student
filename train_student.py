import numpy as np
import timm
import torch
import wandb
from torch import nn
import os
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader


from utils.student_parser import get_parser
from dataset.pytorch_dataset import dataset2_v2
from dataset.augmentations import get_training_augmentations, get_validation_augmentations
from utils.configurators import (
    config_optimizers,
    config_schedulers,
)

# CMD
parser = get_parser()
args = parser.parse_args()


def main():

    # init w&b:
    wandb.init(project=args.project_name, config=vars(args), group=args.group, save_code=False, name=args.name)

    # init models:
    if args.model=='resnet50':
        teacher = timm.create_model('resnet50', pretrained=True, num_classes=1)
        student = timm.create_model('resnet50', pretrained=True, num_classes=1)
    elif args.model=='vit-small':
        teacher = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=1)
        student = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=1)
    elif args.model=='swin-tiny':
        teacher = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=1)
        student = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=1)

    else:
        print('NO model selected')
  
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location='cpu'))
    student.load_state_dict(torch.load(args.teacher_weights, map_location='cpu'))

    teacher = teacher.to(args.device)
    student = student.to(args.device)

    train_transforms = get_training_augmentations('geometric')
    valid_transforms = get_validation_augmentations()

    # set paths for training
    train_dataset = dataset2_v2(args.train_dir, train_transforms)
    valid_dataset = dataset2_v2(args.valid_dir, valid_transforms)

    # define dataloaders
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, num_workers=args.workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # optimizer
    optimizer = config_optimizers(student.parameters(), args)
    scheduler = config_schedulers(optimizer, args)

    # define criterion
    criterion = nn.BCEWithLogitsLoss()

    # checkpointing - directories
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    print(args.save_model_path)

    # define value for min-loss
    min_loss = float("inf")

    print("Training starts...")
    for epoch in range(args.epochs):
        wandb.log({"epoch": epoch})
        train_student(
            student=student,
            teacher=teacher,
            train_dataloader=train_dataloader,
            args=args,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epoch=epoch,
        )
        print('Validating....')
        val_results = validate_epoch(
            student, val_dataloader=valid_dataloader, args=args, criterion=criterion
        )
        if val_results["val_loss"] < min_loss:
            min_loss = val_results["val_loss"].copy()
            ckpt_name = "best-ckpt.pt"
            torch.save(student.state_dict(), os.path.join(args.save_model_path, ckpt_name))


def train_student(teacher, student, train_dataloader, args, optimizer, criterion, scheduler=None, epoch=0, val_results={}):
    student.train()
    teacher.eval()

    epoch += 1
    running_loss = []

    # define loss
    l2_loss = nn.MSELoss()
    # define pooling
    avg_pool = nn.AdaptiveAvgPool2d((1,1))

    pbar = tqdm(train_dataloader, desc=f"epoch {epoch}.", unit="iter")

    for batch, (x, y) in enumerate(pbar):
        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)

        optimizer.zero_grad()


        # representation loss
        # first with spatial info:
        if args.volume_loss:
            if args.model == 'swin-tiny':
                student.features = swin_features.__get__(student)
                teacher.features = swin_features.__get__(teacher)

                student_features = student.features(x)
                teacher_features = teacher.features(x)

            elif args.model == 'vit-small':
                student.features = vit_features.__get__(student)
                teacher.features = vit_features.__get__(teacher)

                student_features = student.features(x)
                teacher_features = teacher.features(x)
                print(student_features.shape)

            elif args.model == 'resnet50':
                student_features = student.forward_features(x)
                teacher_features = teacher.forward_features(x)

            else:
                break
            
            # calculate loss:
            repr_loss = l2_loss(student_features, teacher_features)

        # without spatial info:
        else:
            # forward features
            student_features = student.forward_features(x)
            with torch.no_grad():
                teacher_features = teacher.forward_features(x)

            student_features_pooled = avg_pool(student_features)
            teacher_features_pooled = avg_pool(teacher_features)

            repr_loss = l2_loss(student_features_pooled, teacher_features_pooled)

        # outputs for binary classification
        outputs = student(x)

        loss = args.alpha*repr_loss + (1.0-args.alpha)*criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.detach().cpu().numpy())

        # log mean loss for the last 10 batches:
        if (batch + 1) % 10 == 0:
            wandb.log({'train-step-loss': np.mean(running_loss[-10:])})
            pbar.set_postfix(loss='{:.3f} ({:.3f})'.format(running_loss[-1], np.mean(running_loss)), **val_results)

    # change the position of the scheduler:
    scheduler.step()

    train_loss = np.mean(running_loss)

    wandb.log({'train-epoch-loss': train_loss})

    return train_loss


# define validation logic
@torch.no_grad()
def validate_epoch(model, val_dataloader, args, criterion):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, y in val_dataloader:

        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)

        outputs = model(x)
        loss = criterion(outputs, y)

        # loss calculation over batch
        running_loss.append(loss.cpu().numpy())

        # accuracy calculation over batch
        outputs = torch.sigmoid(outputs)
        outputs = torch.round(outputs)
        y_true.append(y.cpu())
        y_pred.append(outputs.cpu())

    y_true = torch.cat(y_true, 0).numpy()
    y_pred = torch.cat(y_pred, 0).numpy()
    val_loss = np.mean(running_loss)
    wandb.log({'validation-loss': val_loss})
    acc = 100. * np.mean(y_true == y_pred)
    wandb.log({'validation-accuracy': acc})

    return {'val_acc': acc, 'val_loss': val_loss}


def swin_features(self, x):
    x = self.patch_embed(x)
    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    x = self.layers(x)
    x = self.norm(x)
    return x

def vit_features(self, x):
    x = self.patch_embed(x)
    if self.cls_token is not None:
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)
    x = self.blocks(x)
    x = self.norm(x)
    return x

if __name__ == '__main__':
    main()
