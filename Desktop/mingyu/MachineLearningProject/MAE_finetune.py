"""
MAE 사전학습 + 파인튜닝
사전학습된 인코더에 MLP/CNN/LSTM 헤드 붙여서 RUL 예측
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, TensorDataset, DataLoader, SequentialSampler

import sys
sys.path.append('/home/affctiv/Desktop/mingyu/MachineLearningProject')

from load_data import load_cmapss, normalize
from split import make_datasets, sliding_window
from MachineLearningProject.Trainer.Training import train, evaluate
from models.MAE_standard import MAEEncoder, pretrain, extract_features

DATA_PATH  = "/home/affctiv/Desktop/mingyu/CMAPSS/6. Turbofan Engine Degradation Simulation Data Set"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FDS        = ['FD001', 'FD002', 'FD003', 'FD004']

# ── 설정 ──────────────────────────────────
WINDOW     = 30
STRIDE     = 1
VAL_RATIO  = 0.2
BATCH_SIZE = 128
EPOCHS     = 40
N_FEATURES = 17
D_MODEL    = 128
MASK_RATIO = 0.25
# ──────────────────────────────────────────


# ── 헤드 정의 ─────────────────────────────

class MLPHead(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, z):
        return self.net(z).squeeze(1)


class CNNHead(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        # 잠재벡터를 시퀀스로 펼쳐서 CNN 적용
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, z):
        x = z.unsqueeze(1)          # (B, 1, d_model)
        x = self.conv(x).squeeze(-1)  # (B, 16)
        return self.fc(x).squeeze(1)


class LSTMHead(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.lstm = nn.LSTM(d_model, 32, batch_first=True)
        self.fc   = nn.Linear(32, 1)

    def forward(self, z):
        x = z.unsqueeze(1)           # (B, 1, d_model)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


class MAEWithHead(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head    = head

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)

# ──────────────────────────────────────────


HEADS = {
    'mlp' : MLPHead,
    'cnn' : CNNHead,
    'lstm': LSTMHead,
}


def run_pretrain(fd):
    """FD별 MAE 사전학습"""
    train_data, train_label, test_data, _ = load_cmapss(DATA_PATH, fd=fd)
    train_data, _ = normalize(train_data, test_data)
    X_train, _    = sliding_window(train_data, train_label, WINDOW, STRIDE)

    save_path = f"checkpoint_mae_std/encoder_{fd}.pth"
    os.makedirs("checkpoint_mae_std", exist_ok=True)

    print(f"\n  [{fd}] MAE 사전학습")
    pretrain(
        X_train, device=DEVICE, d_model=D_MODEL,
        mask_ratio=MASK_RATIO, epochs=50,
        save_path=save_path
    )


def train_final(fd, head_name):
    """사전학습된 인코더 + 헤드로 파인튜닝"""
    train_data, train_label, test_data, test_label = load_cmapss(DATA_PATH, fd=fd)
    train_data, test_data = normalize(train_data, test_data)

    dataset_train, dataset_val, dataset_test = make_datasets(
        train_data, train_label, test_data, test_label,
        window=WINDOW, stride=STRIDE, val_ratio=VAL_RATIO
    )

    # 인코더 로드
    encoder = MAEEncoder(N_FEATURES, WINDOW, D_MODEL).to(DEVICE)
    ckpt    = f"checkpoint_mae_std/encoder_{fd}.pth"
    encoder.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # 헤드 붙이기
    head  = HEADS[head_name](D_MODEL).to(DEVICE)
    model = MAEWithHead(encoder, head).to(DEVICE)

    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 0.0001},  # 인코더: 낮은 lr (사전학습 보존)
        {'params': model.head.parameters(),    'lr': 0.001},   # 헤드: 높은 lr
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    result = train(
        model=model, dataset_train=dataset_train,
        dataset_val=dataset_val, dataset_test=dataset_test,
        device=DEVICE, output_dir=f"result_mae_{head_name}_{fd}/",
        optimizer=optimizer, scheduler=scheduler,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        criterion=nn.MSELoss()
    )
    return result['rmse'], result['nasa_score']


def main():
    print(f"Device: {DEVICE}\n")

    # Step 1: 사전학습
    print("="*60)
    print("  Step 1: MAE 사전학습 (FD별)")
    print("="*60)
    for fd in FDS:
        run_pretrain(fd)

    # Step 2: 헤드별 파인튜닝
    print("\n" + "="*60)
    print("  Step 2: 헤드별 파인튜닝")
    print("="*60)

    all_results = {}
    for head_name in ['mlp', 'cnn', 'lstm']:
        all_results[head_name] = {}
        print(f"\n[MAE + {head_name.upper()}]")
        for fd in FDS:
            r, s = train_final(fd, head_name)
            all_results[head_name][fd] = {'rmse': r, 'nasa_score': s}

    # 최종 결과 출력
    print("\n" + "#"*60)
    print("  MAE 사전학습 + 헤드별 최종 결과")
    print("#"*60)
    for head_name in ['mlp', 'cnn', 'lstm']:
        print(f"\n  MAE + {head_name.upper()}")
        for fd in FDS:
            res = all_results[head_name][fd]
            print(f"  {fd}: RMSE {res['rmse']:>7.4f}  NASA Score {res['nasa_score']:>12.2f}")
    print("#"*60)


if __name__ == '__main__':
    main()