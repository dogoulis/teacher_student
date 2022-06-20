import numpy as np
import timm
import torch
import wandb
from torch import nn
import os
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader


from utils.teacher_parser import get_parser
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
    wandb.init(project=args.project_name, config=vars(args), group=args.gropu, save_code=False)

    # init model:
    model = timm.create_model('resnet50', pretrained=True, num_classes=1, drop_path_rate=0.1)
    model = model.to(args.device)

    train_transforms = get_training_augmentations()
    valid_transforms = get_validation_augmentations()

    # set paths for training
    train_dataset = dataset2_v2(args.train_dir, train_transforms)
    valid_dataset = dataset2_v2(args.valid_dir, valid_transforms)

    # define dataloaders
    train_dataloader = DataLoader(train_dataset, num_workers=args.worksers, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, num_workers=args.worksers, batch_size=args.batch_size, shuffle=False)

    # optimizer
    optimizer = config_optimizers(model.parameters(), args)
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
        train_teacher(
            model,
            train_dataloader=train_dataloader,
            args=args,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epoch=epoch,
        )
        val_results = validate_epoch(
            model, dataloader=valid_dataloader, args=args, criterion=criterion
        )

        if val_results["val_loss"] < min_loss:
            min_loss = val_results["val_loss"].copy()
            ckpt_name = f"{wandb.run.name}_epoch_{epoch}_val_loss_{val_results['val_loss']:.4f}.pt"
            torch.save(model.state_dict(), os.path.join(args.save_model_path, ckpt_name))


def train_teacher(model, train_dataloader, args, optimizer, criterion, scheduler=None, epoch=0, val_results={}):
    model.train()
    epoch += 1
    running_loss = []

    pbar = tqdm(train_dataloader, desc=f"epoch {epoch}.", unit="iter")

    for batch, (x, y) in enumerate(pbar):
        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

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
