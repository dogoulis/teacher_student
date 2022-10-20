import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import timm
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from dataset import pytorch_dataset, augmentations
from torch.utils.data.dataloader import DataLoader
from torchmetrics import functional as tmf
from utils.test_parser import get_parser
from sklearn.metrics import roc_curve, roc_auc_score

# CMD
parser = get_parser()
args = parser.parse_args()

class GanEnsemble(nn.Module):
    def __init__(self, model_names, num_classes=1, ckpt_path=None):
        super(GanEnsemble, self).__init__()
        self.models = nn.ModuleList()

        for name in model_names:
            self.models.append(timm.create_model(name, num_classes=num_classes))

        # load weights:
        self.models[0].load_state_dict(
            torch.load(os.path.join(ckpt_path, "resnet50.pt"), map_location="cpu",)
        )
        self.models[1].load_state_dict(
            torch.load(os.path.join(ckpt_path, "swin-tiny.pt"), map_location="cpu",)
        )
        self.models[2].load_state_dict(
            torch.load(os.path.join(ckpt_path, "vit-small.pt"), map_location="cpu",)
        )

    def forward(self, x):
        
        res = torch.cat([model(x) for model in self.models], dim=1)
        res = torch.sigmoid(res)
        res = res.mean(dim=1)
                

        return res


@torch.no_grad()
def testing(model, dataloader, criterion):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, y in tqdm(dataloader):
        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)
        outputs = model(x).unsqueeze(1) # if enemble then add .unsqueeze(1)
        loss = criterion(outputs, y) 

        running_loss.append(loss.cpu().numpy())
        outputs = torch.sigmoid(outputs)
        y_true.append(y.squeeze(1).cpu().int())
        y_pred.append(outputs.squeeze(1).cpu())
    wandb.log({'Loss': np.mean(running_loss)})

    return np.mean(running_loss), torch.cat(y_true, 0), torch.cat(y_pred, 0)


def log_metrics(y_true, y_pred):

    test_acc = tmf.accuracy(y_pred, y_true)
    test_f1 = tmf.f1_score(y_pred, y_true)
    test_prec = tmf.precision(y_pred, y_true)
    test_rec = tmf.recall(y_pred, y_true)
    test_auc = tmf.auroc(y_pred, y_true)

    wandb.log({
        'Accuracy': test_acc,
        'F1': test_f1,
        'Precision': test_prec,
        'Recall': test_rec,
        'ROC-AUC score': test_auc})


def log_conf_matrix(y_true, y_pred):
    conf_matrix = tmf.confusion_matrix(y_pred, y_true, num_classes=2)
    conf_matrix = pd.DataFrame(data=conf_matrix, columns=['A', 'B'])
    cf_matrix = wandb.Table(dataframe=conf_matrix)
    wandb.log({'conf_mat': cf_matrix})

def log_plot(y_true, y_pred):
    
    m_probs = y_pred 

    m_fpr, m_tpr, _ = roc_curve(y_pred, m_probs)

    plt.plot(m_fpr, m_tpr, marker='.', label='Model Guess')


# main def:
def main():

    # initialize w&b
    print(args.name)
    wandb.init(project=args.project_name, name=args.name,
               config=vars(args), group=args.group, entity=args.entity)

    # initialize model:
    if args.model == 'resnet50':
        model = timm.create_model('resnet50', num_classes=1, pretrained=True)
    elif args.model == 'swin-tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=1, pretrained=True)
    elif args.model == 'vit-small':
        model = timm.create_model('vit_small_patch16_224', num_classes=1, pretrained=True)
    elif args.model == 'xception':
        model = timm.create_model('xception', num_classes=1, pretrained=True)

    elif args.model == 'ensemble':
        config = {
        'gan_checkpoints':'/fssd1/user-data/dogoulis/networks/',
        'device':'cuda:1'
        }
        gan_model_names = ['resnet50', 'swin_tiny_patch4_window7_224', 'vit_small_patch16_224']
        model = GanEnsemble(gan_model_names, num_classes=1, ckpt_path=config['gan_checkpoints'])
    
    else:
        print('No selected model')
    # load weights:
    if args.student_weights:
        model.load_state_dict(torch.load(args.student_weights, map_location='cpu'))
    if args.teacher_weights:
        model.load_state_dict(torch.load(args.teacher_weights, map_location='cpu'))

    model = model.eval().to(args.device)

    # defining transforms:
    transforms = augmentations.get_validation_augmentations()

    # define test dataset:
    test_dataset = pytorch_dataset.dataset2_v2(args.test_dir, transforms)

    # define data loaders:
    test_dataloader = DataLoader(test_dataset, num_workers=args.workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # set the criterion:
    criterion = nn.BCEWithLogitsLoss()

    # testing
    test_loss, y_true, y_pred = testing(
        model=model, dataloader=test_dataloader, criterion=criterion)

    # calculating and logging results:
    log_metrics(y_true=y_true, y_pred=y_pred)
    log_conf_matrix(y_true=y_true, y_pred=y_pred)

    print(f'Finished Testing with test loss = {test_loss}')


if __name__ == '__main__':
    main()
