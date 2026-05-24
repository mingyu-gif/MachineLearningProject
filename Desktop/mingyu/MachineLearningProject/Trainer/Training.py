"""
CBAM RUL 예측 학습 루프 (회귀 버전)
기존 CBAMTraining.py 구조 유지, 분류→회귀로 변경
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

def nasa_score(y_pred, y_true):
    """
    NASA Score (비대칭 패널티 지표)
    늦게 예측(고장 임박인데 정상으로 예측)하면 더 큰 패널티
    """
    diff  = y_pred - y_true
    score = np.where(
        diff < 0,
        np.exp(-diff / 13) - 1,
        np.exp(diff  / 10) - 1
    )
    return float(np.sum(score))

def rmse(y_pred, y_true):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def train(model, dataset_train, dataset_val, dataset_test,
          device, output_dir="result/",
          optimizer=None, scheduler=None,
          batch_size=128, epochs=40, criterion=None):
          
    os.makedirs(output_dir, exist_ok=True)
    best_path = f"{output_dir}/best_model.pth"

    sampler_train = RandomSampler(dataset_train)
    sampler_val   = SequentialSampler(dataset_val)
    sampler_test  = SequentialSampler(dataset_test)

    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4)
    data_loader_val   = DataLoader(dataset_val,   sampler=sampler_val,   batch_size=batch_size, num_workers=4)
    data_loader_test  = DataLoader(dataset_test,  sampler=sampler_test,  batch_size=batch_size, num_workers=4)

    model       = model.to(device)
    best_rmse   = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(
            data_loader_train,
            desc=f"Epoch [{epoch+1:3d}/{epochs}] lr:{optimizer.param_groups[0]['lr']:.6f}"
        )
        
        for samples, targets in train_bar:
            samples = samples.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(samples)
            loss    = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            train_bar.set_postfix_str(f"loss: {loss.item():.4f}")

        if scheduler is not None:
            scheduler.step()

        val_rmse, val_score = evaluate(model, data_loader_val, device)
        print(f"\033[32m  [Val]  RMSE: {val_rmse:.4f}  NASA Score: {val_score:.2f}\033[0m")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_path)

    # Test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_rmse, test_score = evaluate(model, data_loader_test, device)
    print(f"\n\033[34m  [Test] RMSE: {test_rmse:.4f}  NASA Score: {test_score:.2f}\033[0m")
    
    return {'rmse': test_rmse, 'nasa_score': test_score}

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    
    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)
        all_preds.extend(outputs.cpu().numpy())
        all_targets.extend(targets.numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    return rmse(all_preds, all_targets), nasa_score(all_preds, all_targets)