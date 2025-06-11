import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVOSNet(nn.Module):
    def __init__(self, num_classes, hidden_dim=128):
        super().__init__()
        # Encoder for each frame (accepts RGB)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # <-- 3 input channels for RGB
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128
            * 16
            * 16,  # assuming input H=W=16 after pooling, adjust as needed
            hidden_size=hidden_dim,
            batch_first=True,
        )
        # Decoder
        self.decoder_fc = nn.Linear(hidden_dim, 128 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x)  # [B*T, 128, H, W]
        # Downsample for LSTM (adjust pooling as needed)
        pooled = F.adaptive_avg_pool2d(feats, (16, 16))  # [B*T, 128, 16, 16]
        pooled = pooled.view(B, T, -1)  # [B, T, 128*16*16]
        lstm_out, _ = self.lstm(pooled)  # [B, T, hidden_dim]
        decoded = self.decoder_fc(lstm_out)  # [B, T, 128*16*16]
        decoded = decoded.view(B * T, 128, 16, 16)
        upsampled = F.interpolate(
            decoded, size=(H, W), mode="bilinear", align_corners=False
        )
        out = self.decoder(upsampled)  # [B*T, num_classes, H, W]
        out = out.view(B, T, -1, H, W)
        return out
