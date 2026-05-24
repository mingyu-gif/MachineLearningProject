"""
1D CNN 베이스라인 실험
기존 main_mlp.py 구조 그대로, 모델만 CNN으로 변경
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset

import sys
sys.path.append('/home/affctiv/Desktop/mingyu/MachineLearningProject')

from load_data import load_cmapss, normalize
from split import make_datasets
from models.CNN_Net import CNN_Net
from Trainer.Training import train

DATA_PATH = "/home/affctiv/Desktop/mingyu/CMAPSS/6. Turbofan Engine Degradation Simulation Data Set"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FDS       = ['FD001', 'FD002', 'FD003', 'FD004']

# ── 설정 ──────────────────────────────────
STRIDE     = 1
VAL_RATIO  = 0.2
BATCH_SIZE = 128
EPOCHS     = 40
N_FEATURES = 17
HIDDEN_DIM = 64

WINDOWS = [20, 30]
LRS     = [0.001, 0.0005]
# ──────────────────────────────────────────


def cross_validate(fd, window, lr, k_folds=3, epochs=20):
    train_data, train_label, test_data, test_label = load_cmapss(DATA_PATH, fd=fd)
    train_data, test_data = normalize(train_data, test_data)

    dataset_full, _, _ = make_datasets(
        train_data, train_label, test_data, test_label,
        window=window, stride=STRIDE, val_ratio=0.0
    )

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_full)):
        train_sub = Subset(dataset_full, train_idx)
        val_sub   = Subset(dataset_full, val_idx)

        model     = CNN_Net(N_FEATURES, HIDDEN_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        result = train(
            model=model, dataset_train=train_sub,
            dataset_val=val_sub, dataset_test=val_sub,
            device=DEVICE, output_dir="temp_cnn_cv/",
            optimizer=optimizer, scheduler=scheduler,
            batch_size=BATCH_SIZE, epochs=epochs,
            criterion=nn.MSELoss()
        )
        fold_rmses.append(result['rmse'])
        print(f"  Fold {fold+1}: RMSE {result['rmse']:.4f}")

    return np.mean(fold_rmses)


def train_final_model(fd, window, lr):
    train_data, train_label, test_data, test_label = load_cmapss(DATA_PATH, fd=fd)
    train_data, test_data = normalize(train_data, test_data)

    dataset_train, dataset_val, dataset_test = make_datasets(
        train_data, train_label, test_data, test_label,
        window=window, stride=STRIDE, val_ratio=VAL_RATIO
    )

    model     = CNN_Net(N_FEATURES, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    result = train(
        model=model, dataset_train=dataset_train,
        dataset_val=dataset_val, dataset_test=dataset_test,
        device=DEVICE, output_dir=f"result_cnn_{fd}/",
        optimizer=optimizer, scheduler=scheduler,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        criterion=nn.MSELoss()
    )
    return result['rmse'], result['nasa_score']


def main():
    print(f"Device: {DEVICE}\n")

    print("="*60)
    print("  Step 1: 하이퍼파라미터 튜닝 (FD001 3-Fold CV)")
    print("="*60)

    best_rmse, best_params = float('inf'), {}
    for w in WINDOWS:
        for lr in LRS:
            print(f"\n[CV] window={w}, lr={lr}")
            avg = cross_validate('FD001', w, lr, k_folds=3, epochs=20)
            print(f"  → Mean RMSE: {avg:.4f}")
            if avg < best_rmse:
                best_rmse   = avg
                best_params = {'window': w, 'lr': lr}

    print(f"\n최적: window={best_params['window']}, lr={best_params['lr']}")

    print("\n" + "="*60)
    print("  Step 2: FD001~FD004 최종 학습")
    print("="*60)

    results = {}
    for fd in FDS:
        print(f"\n[{fd}] 학습 중...")
        r, s = train_final_model(fd, best_params['window'], best_params['lr'])
        results[fd] = {'rmse': r, 'nasa_score': s}

    print("\n" + "#"*60)
    print("  CNN 베이스라인 최종 결과")
    print("#"*60)
    for fd, res in results.items():
        print(f"  {fd}: RMSE {res['rmse']:>7.4f}  NASA Score {res['nasa_score']:>12.2f}")
    print("#"*60)


if __name__ == '__main__':
    main()