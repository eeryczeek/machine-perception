import torch
from dataloader import VOSDataset
from model import CNNWithLSTM
from torch.utils.data import DataLoader, Subset
import random

def evaluate(model_path, im_root, gt_root, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNWithLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = VOSDataset(im_root, gt_root, max_jump=1, is_bl=False)
    subset = Subset(dataset, random.sample(range(len(dataset)), int(0.1 * len(dataset))))
    loader = DataLoader(subset, batch_size=16, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for data in loader:
            imgs, labels = data["rgb"].to(device), data["cls_gt"].to(device)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

    return 100 * correct / total