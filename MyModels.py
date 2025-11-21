import torch
import torch.nn as nn
import torchvision.models as models


##############################################################################################
##############################################################################################

import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR_Model(nn.Module):
    def __init__(self, backbone_type, projection_dim=128):
        super(SimCLR_Model, self).__init__()

        # Select backbone
        if backbone_type == "resnet18":
            resnet = models.resnet18(pretrained=False)
        elif backbone_type == "resnet50":
            resnet = models.resnet50(pretrained=False)
        else:
            raise ValueError("Invalid backbone_type. Choose 'resnet18' or 'resnet50'.")

        # Encoder: all layers except the final FC
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Projection head
        in_features = resnet.fc.in_features
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze()  # Feature representation
        z = self.projector(h)          # Projection head output
        return h, z

##############################################################################################
##############################################################################################
import torch
import torch.nn as nn

class LinearEvaluationModel(nn.Module):
    def __init__(self, simclr_model, num_classes=10):
        super(LinearEvaluationModel, self).__init__()
        
        # Freeze encoder parameters
        for param in simclr_model.encoder.parameters():
            param.requires_grad = False

        self.encoder = simclr_model.encoder  


        sample_input = torch.randn(1, 3, 224, 224) 
        with torch.no_grad():
            feature_dim = self.encoder(sample_input).squeeze().shape[-1]


        # Larger classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x).squeeze()  
        logits = self.classifier(h)       
        return logits

##############################################################################################
##############################################################################################




