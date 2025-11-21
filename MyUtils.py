import os
import random
import numpy as np
import torch
import tensorflow as tf



##############################################################################################
##############################################################################################

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)

##############################################################################################
##############################################################################################

import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self._get_correlated_mask().to(device)
        self.similarity_function = self._cosine_similarity

    def _get_correlated_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(False)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = False
            mask[self.batch_size + i, i] = False
        return mask

    def _cosine_similarity(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return torch.matmul(x, y.T)

    def forward(self, z_i, z_j):
        # Concatenate positive pairs
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        sim = self.similarity_function(z, z) / self.temperature
        # Remove similarity between positive pairs from denominator
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)

        # Mask out self-similarities
        sim = sim[self.mask].view(2 * self.batch_size, -1)

        # Compute loss
        #loss = -torch.log(torch.exp(positives) / torch.sum(torch.exp(sim), dim=1))
        loss = -positives + torch.logsumexp(sim, dim=1)
        return loss.mean()



##############################################################################################
##############################################################################################

import os
import matplotlib.pyplot as plt

def plot_and_save_losses(epoch_losses, folder='Plots', name_plot="loss"):

    # Create 'Plots' directory if it doesn't exist
    os.makedirs('Plots', exist_ok=True)

    # Plot the losses
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-', color='blue')
    plt.title(f'{name_plot.capitalize()} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(name_plot.capitalize())
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(folder, f'{name_plot}.png')
    plt.savefig(plot_path)
    plt.close()



##############################################################################################
##############################################################################################

import os
import torchvision.utils as vutils

def save_tensor_image(image_tensor, folder='Plots', filename='image.png'):

    if not os.path.isdir(folder):
        os.mkdir(folder)

    save_path = os.path.join(folder, filename)
    vutils.save_image(image_tensor, save_path)
    print(f"Image saved to {save_path}")



##############################################################################################
##############################################################################################
 
def save_simclr_model(model, path="saved_models/simclr_model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"SimCLR model saved to {path}")

##############################################################################################
##############################################################################################

def load_simclr_model(model, path="saved_models/simclr_model.pth", device='cuda'):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"SimCLR model loaded from {path}")
    return model

##############################################################################################
##############################################################################################

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader


def visualize_tsne(model, dataset, device, save_path="saved_models/tsne_visualization.png", batch_size=256):
    # Ensure dataset is in torch format
    dataset.set_format("torch", columns=["image", "label"])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    embeddings = []
    labels = []

    # Extract embeddings
    with torch.no_grad():
        for batch in data_loader:
            images, lbls = batch["image"].to(device), batch["label"]
            emb = model(images)  
            embeddings.append(emb.cpu())
            labels.append(lbls)

    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(set(labels))))
    plt.title("t-SNE Visualization of Test Embeddings")

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"t-SNE visualization saved at {save_path}")

##############################################################################################
##############################################################################################


import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import os

def visualize_tsne_3d_interactive(model, dataset, device, save_path="saved_models/tsne_3d.html", batch_size=256):
    # Ensure dataset is in torch format
    dataset.set_format("torch", columns=["image", "label"])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    embeddings, labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            images, lbls = batch["image"].to(device), batch["label"]
            emb = model(images)  # Use forward if it returns features
            embeddings.append(emb.cpu())
            labels.append(lbls)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    # Apply t-SNE for 3D
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    embeddings_3d = tsne.fit_transform(embeddings)

    # Create DataFrame for Plotly
    df = pd.DataFrame(embeddings_3d, columns=["x", "y", "z"])
    df["label"] = labels

    # Plot interactive 3D scatter
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="label", title="Interactive 3D t-SNE Visualization")
    
    # Save as HTML
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)

    print(f"Interactive 3D t-SNE visualization saved at {save_path}. Open this file in your browser to rotate and zoom.")

##############################################################################################
##############################################################################################

def Train_SimCLR(model, data_loader, optimizer, scheduler, loss_fn, batch_size, epochs, device, debug=False):
    model.to(device)
    model.train()

    epoch_loss = []
    for epoch in range(epochs):
        total_loss = 0.0

        for step, (x, x_i, x_j) in enumerate(data_loader):
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            # Forward pass
            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)

            # Compute loss
            loss = loss_fn(z_i, z_j)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if debug and step % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(data_loader)}], Loss: {loss.item():.4f}")

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(data_loader)
        epoch_loss.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")
    
    return epoch_loss

##############################################################################################
##############################################################################################

def train_linear_model(model, train_loader, test_loader, optimizer, scheduler, loss_fn, device, epochs):
    epoch_loss = []
    epoch_acc = []
    epoch_test_acc = []

    for epoch in range(epochs):
        model.train()
        batch_loss = []
        correct_train = 0
        total_train = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(float(loss))

            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        avg_loss = np.mean(batch_loss)
        train_acc = 100. * correct_train / total_train
        epoch_loss.append(avg_loss)
        epoch_acc.append(train_acc)

        # Evaluation
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)
                correct_test += predicted.eq(labels).sum().item()
                total_test += labels.size(0)

        test_acc = 100. * correct_test / total_test
        epoch_test_acc.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return epoch_loss, epoch_acc, epoch_test_acc

##############################################################################################
##############################################################################################





