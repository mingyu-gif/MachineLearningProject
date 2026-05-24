"""
슬라이딩 윈도우 기반 시퀀스 생성 및 데이터 분할
LibEER split.py 구조 참고하여 구현
"""
import numpy as np
from torch.utils.data import TensorDataset
import torch

def sliding_window(data, label, window=30, stride=1):
    """
    슬라이딩 윈도우로 시퀀스 생성
    data  : (N, n_features) — 전체 시계열
    label : (N,) — RUL 값
    window: 윈도우 크기 (몇 사이클을 보고 예측할지)
    stride: 슬라이딩 간격
    반환:
      X: (N_seq, window, n_features)
      y: (N_seq,) — 윈도우 마지막 시점의 RUL
    """
    X_list, y_list = [], []
    N = len(data)
    
    for i in range(0, N - window + 1, stride):
        X_list.append(data[i:i+window])
        y_list.append(label[i+window-1])  # 마지막 시점 RUL
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def make_datasets(train_data, train_label, test_data, test_label,
                  window=30, stride=1, val_ratio=0.2):
    """
    train/val/test TensorDataset 생성
    val_ratio: train에서 val로 분리할 비율
    """
    # 슬라이딩 윈도우 적용
    X_train, y_train = sliding_window(train_data, train_label, window, stride)
    X_test,  y_test  = sliding_window(test_data,  test_label,  window, stride)

    # train → train + val 분리
    n_val   = int(len(X_train) * val_ratio)
    n_train = len(X_train) - n_val
    X_val,   y_val   = X_train[n_train:], y_train[n_train:]
    X_train, y_train = X_train[:n_train], y_train[:n_train]

    print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    # TensorDataset 변환
    dataset_train = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    dataset_val = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )
    dataset_test = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )
    
    return dataset_train, dataset_val, dataset_test