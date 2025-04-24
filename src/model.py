import torch
import torch.nn as nn
from torchvision import models


class CNNWithLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNWithLSTM, self).__init__()
        self.cnn = models.mobilenet_v2()
        self.cnn.classifier[1] = nn.Identity()  # Remove the final classification layer
        self.lstm = nn.LSTM(
            input_size=1280, hidden_size=512, num_layers=1, batch_first=True
        )
        self.conv3d = nn.Conv3d(
            in_channels=512, out_channels=num_classes, kernel_size=(3, 3, 3), padding=1
        )  # 3D convolution for segmentation output

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(
            batch_size * num_frames, channels, height, width
        )  # Flatten temporal dimension
        x = self.cnn(x)  # Pass through CNN
        x = x.view(batch_size, num_frames, -1)  # Reshape for LSTM
        x, _ = self.lstm(x)  # Pass through LSTM
        x = x.unsqueeze(2)  # Add a depth dimension for 3D convolution
        x = self.conv3d(x)  # Pass through 3D convolution
        return x
