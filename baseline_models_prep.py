import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vit_b_16
try:
    # torchvision >= 0.14
    from torchvision.models import ViT_B_16_Weights
except Exception:
    ViT_B_16_Weights = None

class LandmarkModel(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, num_landmarks=5):
        super(LandmarkModel, self).__init__()
        self.backbone_type = backbone
        self.num_outputs = num_landmarks * 2

        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.num_outputs)

        elif backbone == 'vit':
            if ViT_B_16_Weights is not None:
                weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = vit_b_16(weights=weights)
            else:
                # Older torchvision versions use 'pretrained' arg
                self.backbone = vit_b_16(pretrained=pretrained)

            # ViT uses heads.head as the classification head
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, self.num_outputs)
        
        else:
            raise ValueError(f"unsupported backbone:{backbone}.")
        
    def forward(self, x):
        return self.backbone(x)
