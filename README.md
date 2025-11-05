SimCLR Replication on CIFAR-10 with ResNet-18
This project replicates the core ideas of the paper "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR) by Chen et al., using the CIFAR-10 dataset and a ResNet-18 backbone.
📌 Overview
SimCLR is a contrastive learning framework that learns visual representations by maximizing agreement between differently augmented views of the same image. This implementation follows the original paper's methodology, adapted for CIFAR-10 and a smaller ResNet-18 model.
🧠 Key Features

Contrastive Learning using NT-Xent loss
SimCLR-style Data Augmentation: Random crop, color jitter, grayscale, Gaussian blur
Projection Head: 2-layer MLP with ReLU
LARS Optimizer for large-batch training
Cosine Annealing Scheduler
Linear Evaluation Protocol: Train a linear classifier on frozen encoder features

📁 Project Structure

main.py: Main training and evaluation script
Config.py: Argument parser and configuration settings
MyDatasets.py: Dataset loading and SimCLR-specific transformations
MyModels.py: Encoder and projection head definitions
MyUtils.py: Training loops, loss functions, plotting, and model I/O

🧪 Training Procedure

Pretraining: SimCLR contrastive learning on CIFAR-10 with augmented views
Evaluation: Freeze encoder and train a linear classifier on top
Metrics: Accuracy and loss plots for both contrastive and linear evaluation phases

⚙️ Requirements

Python 3.8+
PyTorch
torchvision
transformers
pl_bolts
matplotlib
numpy
