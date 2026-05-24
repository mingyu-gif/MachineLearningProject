"""
LSTM 베이스라인
입력: (B, window, n_features)
출력: (B,) RUL
"""

import torch
import torch.nn as nn


class LSTM_Net(nn.Module):
    def __init__(self, n_features=17, hidden_dim=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, window, n_features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # 마지막 타임스텝
        return self.fc(out).squeeze(1)