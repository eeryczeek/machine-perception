import os
from dataloader import VOSDataset

if __name__ == "__main__":
    # Use os.path.join to construct paths
    im_root = os.path.join("data", "train", "JPEGImages")
    gt_root = os.path.join("data", "train", "Annotations")

    dataset = VOSDataset(im_root, gt_root, 1, False)
    print("Dataset loaded successfully.")
    print(f"Number of videos: {len(dataset)}")
