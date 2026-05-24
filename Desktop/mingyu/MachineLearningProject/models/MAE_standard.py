"""
표준 MAE (Masked Autoencoder)
- 시간 스텝 단위 마스킹
- Transformer 인코더
- Linear 디코더
- EEG MAE와 동일한 구조, 도메인만 엔진으로 변경
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════
# 1. 인코더 (Transformer)
# ══════════════════════════════════════════

class MAEEncoder(nn.Module):
    """
    Transformer 기반 인코더
    입력: (B, window, n_features)
    출력: (B, d_model) 잠재벡터
    """
    def __init__(self, n_features=17, window=30, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        # Learnable Mask Token
        self.mask_token = nn.Parameter(torch.randn(n_features))

        # 입력 투영
        self.input_proj = nn.Linear(n_features, d_model)

        # 위치 임베딩
        self.pos_emb = nn.Parameter(torch.randn(1, window, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, window, n_features)
        x = self.input_proj(x)    # (B, window, d_model)
        x = x + self.pos_emb
        x = self.transformer(x)
        x = self.norm(x)
        return x.mean(dim=1)      # (B, d_model)


# ══════════════════════════════════════════
# 2. 디코더 (Linear)
# ══════════════════════════════════════════

class MAEDecoder(nn.Module):
    """
    Linear 디코더
    잠재벡터 → 원본 시계열 복원
    """
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
# 3. 시간 축 마스킹
# ══════════════════════════════════════════

def time_masking(x, encoder, mask_ratio=0.25):
    """
    시간 스텝 단위 마스킹
    x: (B, window, n_features)
    반환: x_masked, mask (B, window) bool
    """
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

def pretrain(X_train, device, d_model=128, nhead=4, num_layers=2,
             mask_ratio=0.25, epochs=50, batch_size=256, lr=1e-3,
             save_path='checkpoint_mae/encoder.pth'):
    """
    MAE 사전학습 후 인코더 가중치 저장
    X_train: (N, window, n_features) numpy
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_features = X_train.shape[2]
    window     = X_train.shape[1]

    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    encoder = MAEEncoder(n_features, window, d_model, nhead, num_layers).to(device)
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

            # 마스킹된 시간 스텝만 loss
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
    """
    사전학습된 인코더로 특징 추출
    반환: (N, d_model)
    """
    encoder.eval()
    dataset = TensorDataset(torch.FloatTensor(X))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats   = []
    for (x,) in loader:
        z = encoder(x.to(device))
        feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)