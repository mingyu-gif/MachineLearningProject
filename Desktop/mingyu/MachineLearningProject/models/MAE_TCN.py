"""
TCN 기반 MAE
- Dilated Causal Convolution 인코더
- 시간 스텝 단위 마스킹
- Linear 디코더
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════
# 1. TCN 블록
# ══════════════════════════════════════════

class TCNBlock(nn.Module):
    """
    Dilated Causal Conv 블록
    dilation으로 넓은 시간 범위 커버
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        # 인과적 패딩 (미래 정보 안 봄)
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1     = nn.BatchNorm1d(out_channels)
        self.bn2     = nn.BatchNorm1d(out_channels)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual 연결 (채널 다르면 1x1 conv)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: (B, channels, time)
        residual = self.residual(x)

        out = self.conv1(x)[:, :, :-self.conv1.padding[0]]  # 인과적 자르기
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)[:, :, :-self.conv2.padding[0]]
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    TCN 기반 인코더
    입력: (B, window, n_features)
    출력: (B, d_model) 잠재벡터
    """
    def __init__(self, n_features=17, d_model=128, num_layers=4, kernel_size=3, dropout=0.1):
        super().__init__()

        # Learnable Mask Token
        self.mask_token = nn.Parameter(torch.randn(n_features))

        # TCN 레이어 (dilation: 1, 2, 4, 8)
        layers = []
        in_ch  = n_features
        for i in range(num_layers):
            dilation = 2 ** i
            out_ch   = d_model
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn  = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, window, n_features)
        x = x.permute(0, 2, 1)      # (B, n_features, window)
        x = self.tcn(x)              # (B, d_model, window)
        x = x.permute(0, 2, 1)      # (B, window, d_model)
        x = self.norm(x)
        return x.mean(dim=1)         # (B, d_model)


# ══════════════════════════════════════════
# 2. 디코더
# ══════════════════════════════════════════

class MAEDecoder(nn.Module):
    def __init__(self, d_model=128, window=30, n_features=17):
        super().__init__()
        self.window     = window
        self.n_features = n_features
        self.decoder    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, window * n_features)
        )

    def forward(self, z):
        out = self.decoder(z)
        return out.view(-1, self.window, self.n_features)


# ══════════════════════════════════════════
# 3. 마스킹
# ══════════════════════════════════════════

def time_masking(x, encoder, mask_ratio=0.25):
    B, T, F  = x.shape
    n_mask   = max(1, int(T * mask_ratio))
    mask     = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    x_masked = x.clone()
    for i in range(B):
        idx = torch.randperm(T, device=x.device)[:n_mask]
        mask[i, idx] = True
        x_masked[i, idx, :] = encoder.mask_token.detach()
    return x_masked, mask


# ══════════════════════════════════════════
# 4. 사전학습
# ══════════════════════════════════════════

def pretrain(X_train, device, d_model=128, num_layers=4, kernel_size=3,
             mask_ratio=0.25, epochs=50, batch_size=256, lr=1e-3,
             save_path='checkpoint_mae_tcn/encoder.pth'):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_features = X_train.shape[2]
    window     = X_train.shape[1]

    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    encoder = TCNEncoder(n_features, d_model, num_layers, kernel_size).to(device)
    decoder = MAEDecoder(d_model, window, n_features).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')

    for epoch in range(epochs):
        encoder.train(); decoder.train()
        epoch_loss = 0.0

        for (x,) in loader:
            x        = x.to(device)
            x_masked, mask = time_masking(x, encoder, mask_ratio)

            z    = encoder(x_masked)
            pred = decoder(z)

            loss  = torch.tensor(0.0, device=device)
            count = 0
            for t in range(x.shape[1]):
                m = mask[:, t]
                if m.sum() == 0:
                    continue
                loss  = loss + F.mse_loss(pred[m, t, :], x[m, t, :])
                count += 1
            loss = loss / max(count, 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0
            )
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        print(f"  Epoch [{epoch+1:3d}/{epochs}]  Loss: {avg:.5f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(encoder.state_dict(), save_path)

    print(f"  완료  best_loss: {best_loss:.5f}  저장: {save_path}")
    encoder.load_state_dict(torch.load(save_path, map_location=device))
    return encoder


# ══════════════════════════════════════════
# 5. 특징 추출
# ══════════════════════════════════════════

@torch.no_grad()
def extract_features(encoder, X, device, batch_size=256):
    encoder.eval()
    dataset = TensorDataset(torch.FloatTensor(X))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats   = []
    for (x,) in loader:
        z = encoder(x.to(device))
        feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)