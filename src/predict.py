import torch
from model import CNNWithLSTM
from dataloader import VOSDataset
from torch.utils.data import DataLoader

# Load model
model = CNNWithLSTM()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Load data
im_root = "data/train/JPEGImages"
gt_root = "data/train/Annotations"
dataset = VOSDataset(im_root, gt_root, max_jump=1, is_bl=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Dodałem plik tylko żeby zobaczyć czy da się model wczytać dw potem użyjemy pewnie
with torch.no_grad():
    sample = next(iter(dataloader))
    input_tensor = sample["rgb"]
    output = model(input_tensor)
    prediction = (torch.sigmoid(output) > 0.5).float()
    prediction_np = prediction.squeeze().cpu().numpy()
    print("Prediction (as numpy array):")
    print(prediction_np)