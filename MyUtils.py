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
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
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
    model.train()
    epoch_loss = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100. * correct / total:.2f}%")


##############################################################################################
##############################################################################################


def train_linear_model(model, train_loader, test_loader, optimizer, scheduler, loss_fn, device, epochs):
    epoch_loss = []
    Acc_train = []
    Acc_test = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100. * correct / total
        epoch_loss.append(total_loss)
        Acc_train.append(train_acc)

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
        Acc_test.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return epoch_loss, Acc_train, Acc_test


##############################################################################################
##############################################################################################





