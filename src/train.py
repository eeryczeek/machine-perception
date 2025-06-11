import os
import numpy as np
import torch
import random
import time
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataloader import VOSDataset
from model import SimpleVOSNet


import random
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont


import torch
import torch.nn as nn
import torch.nn.functional as F


class JaccardLossNoBackground(nn.Module):
    def __init__(self, ignore_index=0, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        # logits: [B, C, H, W], target: [B, H, W]
        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(
                logits, size=target.shape[1:], mode="bilinear", align_corners=False
            )
        num_classes = logits.shape[1]
        fg_logits = logits  # [B, C-1, H, W]
        fg_target = target.clone()
        fg_target[target == self.ignore_index] = 0
        fg_target = F.one_hot(fg_target, num_classes=num_classes)
        fg_target = fg_target.permute(0, 3, 1, 2).float()  # [B, C-1, H, W]
        fg_probs = torch.softmax(fg_logits, dim=1)
        # Ensure shapes match
        assert fg_probs.shape == fg_target.shape, f"{fg_probs.shape}, {fg_target.shape}"
        intersection = torch.sum(fg_probs * fg_target, dim=(0, 2, 3))
        union = torch.sum(fg_probs + fg_target, dim=(0, 2, 3)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        loss = 1 - iou.mean()
        return loss


@torch.no_grad()
def make_gif_from_random_val_clip(
    model, val_dataset, save_dir="output_gif", device="cuda"
):
    os.makedirs(save_dir, exist_ok=True)
    # Pick a random clip index from the validation set
    idx = random.randint(0, len(val_dataset) - 1)
    sample = val_dataset[idx]
    # sample["rgb"]: [T, 3, H, W], sample["cls_gt"]: [T, H, W]
    rgb = sample["rgb"].to(device)  # [T, 3, H, W]
    cls_gt = sample["cls_gt"]  # [T, H, W]
    T = rgb.shape[0]
    font = ImageFont.load_default()
    frames = []

    # Model expects [B, T, 3, H, W]
    rgb_input = rgb.unsqueeze(0)  # [1, T, 3, H, W]
    logits = model(rgb_input)  # [1, T, C, H, W]
    preds = torch.argmax(logits, dim=2).squeeze(0).cpu()  # [T, H, W]

    for t in range(T):
        img = to_pil_image(rgb[t].cpu())
        gt_frame = cls_gt[t]
        if hasattr(gt_frame, "cpu"):
            gt_frame = gt_frame.cpu().numpy()
        gt_frame = np.squeeze(gt_frame)  # Ensure shape is (H, W)
        anno = Image.fromarray(gt_frame.astype(np.uint8)).convert("P")
        pred = Image.fromarray(preds[t].numpy().astype(np.uint8)).convert("P")

        # Colorize masks for visualization
        anno = anno.convert("RGB")
        pred = pred.convert("RGB")

        # Draw labels
        draw_anno = ImageDraw.Draw(anno)
        draw_pred = ImageDraw.Draw(pred)
        draw_anno.text((10, 10), "Original", fill="white", font=font)
        draw_pred.text((10, 10), "Predicted", fill="white", font=font)

        # Compose frame
        w, h = anno.width + pred.width, anno.height
        combined = Image.new("RGB", (w, h))
        combined.paste(anno, (0, 0))
        combined.paste(pred, (pred.width, 0))
        frames.append(combined)

    gif_path = os.path.join(save_dir, f"val_clip_{idx}.gif")
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0
    )
    print(f"Validation GIF saved at {gif_path}")


import torch
import torch.nn.functional as F


def multiclass_dice_loss(input, target, epsilon=1e-6):
    # input: [B, C, H, W], target: [B, H, W]
    input = F.softmax(input, dim=1)
    target_onehot = (
        F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
    )
    dims = (0, 2, 3)
    intersection = torch.sum(input * target_onehot, dims)
    cardinality = torch.sum(input + target_onehot, dims)
    dice = (2.0 * intersection / (cardinality + epsilon)).mean()
    return 1 - dice


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    total_samples = len(dataloader.dataset)
    trained_samples = 0
    start_time = time.time()

    for batch_idx, data in enumerate(dataloader):
        images, labels = data["rgb"].to(device), data["cls_gt"].to(
            device
        )  # images: [B,T,3,H,W], labels: [B,T,H,W]
        outputs = model(images)  # [B,T,C,H,W]
        # reshape for loss: merge batch and time dims
        B, T, C, H, W = outputs.shape
        outputs = outputs.view(B * T, C, H, W)
        labels = labels.view(B * T, H, W).long()
        if outputs.shape[2:] != labels.shape[1:]:
            outputs = F.interpolate(
                outputs, size=labels.shape[1:], mode="bilinear", align_corners=False
            )
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
        eta_total = eta_epoch + avg_batch_time * batches_total * (
            num_epochs - epoch - 1
        )

        print(
            f"Batch {batches_done}/{batches_total} - "
            f"Trained {trained_samples}/{total_samples} samples - "
            f"ETA (epoch): {int(eta_epoch)}s - "
            f"ETA (total): {int(eta_total)}s"
        )

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
            B, T, C, H, W = outputs.shape
            outputs = outputs.view(B * T, C, H, W)
            labels = labels.view(B * T, H, W).long()
            if outputs.shape[2:] != labels.shape[1:]:
                outputs = F.interpolate(
                    outputs, size=labels.shape[1:], mode="bilinear", align_corners=False
                )
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)  # [B*T, H, W]
            correct += (preds == labels).sum().item()
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
    num_objects = 35  # number of objects (excluding background)
    num_classes = num_objects  # +1 for background

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_full = VOSDataset(im_root, gt_root, max_jump=1, is_bl=False)
    dataset_size = 0.1  # % of dataset to use
    subset_size = int(dataset_size * len(train_dataset_full))
    subset_indices = random.sample(range(len(train_dataset_full)), subset_size)

    train_dataset = Subset(train_dataset_full, subset_indices)
    val_dataset = Subset(train_dataset_full, subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleVOSNet(num_classes=num_objects).to(device)
    criterion = JaccardLossNoBackground(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        make_gif_from_random_val_clip(model, val_dataset)

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
