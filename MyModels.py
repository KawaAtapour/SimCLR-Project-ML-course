import torch
import torch.nn as nn
import torchvision.models as models


##############################################################################################
##############################################################################################

class SimCLR_Model(nn.Module):
    def __init__(self, projection_dim=128):
        super(SimCLR_Model, self).__init__()
        
        resnet = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  

        self.projector = nn.Sequential(
            nn.Linear(resnet.fc.in_features, resnet.fc.in_features),
            nn.ReLU(),
            nn.Linear(resnet.fc.in_features, projection_dim)
        )


    def forward(self, x):
        h = self.encoder(x).squeeze() 
        z = self.projector(h)         
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

        # Larger classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
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




