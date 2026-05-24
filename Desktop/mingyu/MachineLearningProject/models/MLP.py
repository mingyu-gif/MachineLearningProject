"""
단순 베이스라인: Multi-Layer Perceptron (MLP)
시계열의 순서를 무시하고 데이터를 1차원으로 펴서 학습하는 단순 회귀 모델
"""
import torch
import torch.nn as nn

class MLP_Net(nn.Module):
    def __init__(self, window=30, n_features=17):
        super(MLP_Net, self).__init__()
        
        # 30 (window) * 17 (features) = 510
        input_dim = window * n_features
        
        self.flatten = nn.Flatten()
        
        # 단순 3층 Feed-Forward 신경망
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        
        # 출력: RUL (잔여 수명 1개)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        # 입력 x: (Batch, Window, Features) -> (Batch, 30, 14)
        x = self.flatten(x)  # (Batch, 420)으로 평탄화
        
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.relu3(self.fc3(x))
        
        out = self.out(x)
        return out.squeeze(1)