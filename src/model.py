import torch
import torch.nn as nn
from torchvision import models


class CNNWithLSTM(nn.Module):
    def __init__(self):
        super(CNNWithLSTM, self).__init__()
        self.cnn = models.mobilenet_v2()
        self.cnn.classifier[1] = nn.Identity()  # Remove the final classification layer
        self.lstm = nn.LSTM(
            input_size=1280, hidden_size=512, num_layers=1, batch_first=True
        )
        self.conv3d = nn.Conv3d(
            in_channels=512, out_channels=3, kernel_size=(3, 3, 3), padding=1
        )  # 3D convolution for RGB output
        self.final_conv = nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=1
        )  # Final 2D convolution to adjust channels

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.size()
        x = x.permute(
            0, 2, 1, 3, 4
        )  # Rearrange to [batch, frames, channels, height, width]
        x = x.reshape(
            batch_size * num_frames, channels, height, width
        )  # Flatten temporal dimension
        x = self.cnn(x)  # Pass through CNN
        x = x.reshape(batch_size, num_frames, -1)  # Reshape for LSTM
        x, _ = self.lstm(x)  # Pass through LSTM
        x = x[:, -1, :]  # Take the output of the last frame
        x = x.reshape(batch_size, 512, 1, 1, 1)  # Reshape for 3D convolution
        x = self.conv3d(x)  # Pass through 3D convolution
        x = x.squeeze(2)  # Remove the temporal dimension
        x = self.final_conv(x)  # Adjust channels with 2D convolution
        x = nn.functional.interpolate(
            x, size=(384, 384), mode="bilinear", align_corners=False
        )  # Upsample to target size
        x = x.unsqueeze(2)  # Add back the temporal dimension as 1
        return x
