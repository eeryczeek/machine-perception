import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from dataloader import VOSDataset
from model import CNNWithLSTM


def dice_loss(outputs, targets, smooth=1e-6):
    """
    Dice Loss for image segmentation.
    Args:
        outputs (torch.Tensor): Predicted mask of shape [batch_size, channels, height, width].
        targets (torch.Tensor): Ground truth mask of shape [batch_size, channels, height, width].
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        torch.Tensor: Computed Dice Loss.
    """
    print(f"outputs shape: {outputs.shape}, targets shape: {targets.shape}")
    outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
    intersection = (outputs * targets).sum(dim=(2, 3))  # Sum over height and width
    union = outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        images, labels = data["rgb"].to(device), data["cls_gt"].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data["rgb"].to(device), data["cls_gt"].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
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
    num_classes = 36  # Replace with the actual number of classes in your dataset

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    train_dataset = VOSDataset(im_root, gt_root, max_jump=1, is_bl=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = VOSDataset(
        im_root, gt_root, max_jump=1, is_bl=False
    )  # Replace with validation dataset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model (using a pre-trained MobileNetV2 as an example)
    model = CNNWithLSTM()
    model = model.to(device)

    # Loss and optimizer
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")
