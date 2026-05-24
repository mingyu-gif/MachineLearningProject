"""
1D CNN 베이스라인
입력: (B, window, n_features)
출력: (B,) RUL
"""

import torch
import torch.nn as nn


class CNN_Net(nn.Module):
    def __init__(self, n_features=17, hidden_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (B, hidden_dim, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, window, n_features)
        x = x.permute(0, 2, 1)      # (B, n_features, window)
        x = self.conv(x)             # (B, hidden_dim, 1)
        x = x.squeeze(-1)            # (B, hidden_dim)
        return self.fc(x).squeeze(1) # (B,)