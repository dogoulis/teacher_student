import torch
from torch import nn
import timm
import os


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