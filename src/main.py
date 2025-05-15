import os
from dataloader import VOSDataset
from evaluate import evaluate

if __name__ == "__main__":
    im_root = os.path.join("data", "train", "JPEGImages")
    gt_root = os.path.join("data", "train", "Annotations")

    dataset = VOSDataset(im_root, gt_root, 1, False)
    print("Dataset loaded successfully.")
    print(f"Number of videos: {len(dataset)}")

    acc = evaluate("model.pth", im_root, gt_root)
    print(f"Validation Accuracy: {acc:.2f}%")