import os
import torch
import random
import time
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
from dataloader import VOSDataset
from model import CNNWithLSTM


def dice_loss(outputs, targets, smooth=1e-6):
    outputs = torch.sigmoid(outputs)
    intersection = (outputs * targets).sum(dim=(2, 3))
    union = outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    total_samples = len(dataloader.dataset)
    trained_samples = 0
    start_time = time.time()

    for batch_idx, data in enumerate(dataloader):
        images, labels = data["rgb"].to(device), data["cls_gt"].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        trained_samples += images.size(0)

        elapsed = time.time() - start_time
        batches_done = batch_idx + 1
        batches_total = len(dataloader)
        avg_batch_time = elapsed / batches_done
        eta_epoch = avg_batch_time * (batches_total - batches_done)
        eta_total = eta_epoch + avg_batch_time * batches_total * (num_epochs - epoch - 1)

        print(f"Batch {batches_done}/{batches_total} - "
              f"Trained {trained_samples}/{total_samples} samples - "
              f"ETA (epoch): {int(eta_epoch)}s - "
              f"ETA (total): {int(eta_total)}s")

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data["rgb"].to(device), data["cls_gt"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy


if __name__ == "__main__":

    # Paths
    im_root = os.path.join("data", "train", "JPEGImages")
    gt_root = os.path.join("data", "train", "Annotations")

    # Hyperparameters
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 36

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset_full = VOSDataset(im_root, gt_root, max_jump=1, is_bl=False)
    dataset_size = 0.01  # % of dataset to use
    subset_size = int(dataset_size * len(train_dataset_full))
    subset_indices = random.sample(range(len(train_dataset_full)), subset_size)

    train_dataset = Subset(train_dataset_full, subset_indices)
    val_dataset = Subset(train_dataset_full, subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNNWithLSTM().to(device)
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    torch.save(model.state_dict(), "model.pth")
    with open("training_history.json", "w") as f:
        json.dump(history, f)
    print("Model saved as model.pth")
    print("Training history saved as training_history.json")