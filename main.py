import torch
import transformers
import numpy as np
import random
import matplotlib.pyplot as plt
from Config import args
import MyUtils
import MyDatasets
import MyModels
import torchvision
import time
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=None)  # or weights=ResNet18_Weights.DEFAULT
import torch.optim as optim
from pl_bolts.optimizers import LARS
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader



dataset_loaders = {
    "cifar10": MyDatasets.load_cifar10,
}


if __name__ == "__main__":
    
    MyUtils.set_seed(42)

    loader = dataset_loaders.get(args.dataset.lower())
    if loader is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")


    base_dataset, num_classes, name_classes = loader(args.num_train_samples, args.num_test_samples)
    simclr_transform = MyDatasets.SimCLRTransform(image_size=32)
    simclr_dataset = MyDatasets.SimCLRDataset(base_dataset['train'], transform=simclr_transform)
    data_loader = DataLoader(simclr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # for x, xi, xj in data_loader:
    #     MyUtils.save_tensor_image(x[-1], folder='Plots', filename='image.png')
    #     MyUtils.save_tensor_image(xi[-1], folder='Plots', filename='image1.png') 
    #     MyUtils.save_tensor_image(xj[-1], folder='Plots', filename='image2.png')
    #     9/0



    model = MyModels.SimCLR_Model()
    base_optimizer = optim.SGD(model.parameters(), lr=4.8, momentum=0.9, weight_decay=1e-6)
    optimizer = LARS( model.parameters(), lr=4.8, momentum=0.9, weight_decay=1e-6, trust_coefficient=0.001 )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    loss_fn = MyUtils.NTXentLoss(args.batch_size)

    epoch_losses = MyUtils.Train_SimCLR(model, data_loader, optimizer, scheduler, loss_fn, args.batch_size, args.epochs, args.device, args.debug)
    
    
    
    
    MyUtils.plot_and_save_losses(epoch_losses, name_plot="loss")



    # ==========================================================================================================================
    # ==========================================================================================================================
    # ==========================================================================================================================

    train_loader = DataLoader(base_dataset['train'], batch_size=args.batch_size_eval, shuffle=True)
    test_loader = DataLoader(base_dataset['test'], batch_size=args.batch_size_eval, shuffle=False)


    eval_model = MyModels.LinearEvaluationModel(model, num_classes).to(args.device)
    eval_optimizer = optim.SGD(eval_model.classifier.parameters(), lr=0.01, momentum=0.9)
    eval_scheduler = CosineAnnealingLR(eval_optimizer, T_max=args.epochs_eval)
    eval_loss_fn = torch.nn.CrossEntropyLoss()


    epoch_loss, Acc_train, Acc_test = MyUtils.train_linear_model(eval_model, train_loader, test_loader, eval_optimizer, eval_scheduler, eval_loss_fn, args.device, args.epochs_eval)

    MyUtils.plot_and_save_losses(epoch_loss, name_plot="loss2")
    MyUtils.plot_and_save_losses(Acc_test, name_plot="Acc_test")
    MyUtils.plot_and_save_losses(Acc_train, name_plot="Acc_train")




