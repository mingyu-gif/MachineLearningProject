# Turbofan Engine RUL Prediction via MAE Pre-training

항공기 터보팬 엔진 잔여 유효 수명(RUL) 예측을 위한 Masked Autoencoder 사전학습 기반 프레임워크

## 프로젝트 개요

NASA C-MAPSS 데이터셋을 사용하여 엔진 센서 시계열에 MAE 사전학습을 적용하고,
다양한 베이스라인(MLP, CNN, LSTM)과 파인튜닝 헤드(MLP, CNN, LSTM)를 비교 분석합니다.

## 파일 구조

```
MachineLearningProject/
├── load_data.py              # C-MAPSS 데이터 로드 및 RUL 레이블 생성
├── split.py                  # 슬라이딩 윈도우 및 데이터 분할
│
├── models/
│   ├── MLP.py                # MLP 베이스라인
│   ├── CNN_Net.py            # 1D CNN 베이스라인
│   ├── LSTM_Net.py           # LSTM 베이스라인
│   ├── MAE_standard.py       # Transformer 기반 MAE 인코더
│   └── MAE_TCN.py            # TCN 기반 MAE 인코더
│
├── Trainer/
│   └── Training.py           # 학습 루프 (RMSE + NASA Score)
│
├── main_mlp.py               # MLP 베이스라인 실험
├── main_cnn.py               # CNN 베이스라인 실험
├── main_lstm.py              # LSTM 베이스라인 실험
├── MAE_finetune.py           # Transformer MAE + 헤드별 실험
└── MAE_TCN_finetune.py       # TCN MAE + 헤드별 실험
```

## 데이터셋

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

```bash
# 데이터 다운로드
wget "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" -O CMAPSS.zip
unzip CMAPSS.zip
```

다운로드 후 `load_data.py`의 `DATA_PATH`를 데이터 경로로 수정하세요.

## 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

### 1. 베이스라인 실험

```bash
# MLP
python main_mlp.py 2>&1 | tee log_mlp.txt

# CNN
python main_cnn.py 2>&1 | tee log_cnn.txt

# LSTM
python main_lstm.py 2>&1 | tee log_lstm.txt
```

### 2. MAE 사전학습 + 파인튜닝

```bash
# Transformer MAE + MLP/CNN/LSTM 헤드
python MAE_finetune.py 2>&1 | tee log_mae_finetune.txt

# TCN MAE + MLP/CNN/LSTM 헤드
python MAE_TCN_finetune.py 2>&1 | tee log_tcn_finetune.txt
```

### 3. 전체 실험 한 번에 실행

```bash
python main_mlp.py && \
python main_cnn.py && \
python main_lstm.py && \
python MAE_finetune.py && \
python MAE_TCN_finetune.py \
2>&1 | tee log_all.txt
```

## 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Window Size | 30 | 슬라이딩 윈도우 크기 (3-Fold CV 튜닝) |
| Stride | 1 | 슬라이딩 간격 |
| Learning Rate | 0.001 | 3-Fold CV 튜닝 결과 |
| Batch Size | 128 | 파인튜닝 / 256 사전학습 |
| MAE Epochs | 50 | 사전학습 반복 횟수 |
| Fine-tune Epochs | 40 | 파인튜닝 반복 횟수 |
| Mask Ratio | 0.25 | MAE 마스킹 비율 |
| d_model | 128 | Transformer 잠재 차원 |
| Max RUL | 125 | Piece-wise Linear 클리핑 값 |

## 실험 결과 요약

### 베이스라인

| 모델 | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|------|-----------|-----------|-----------|-----------|
| MLP  | 16.17 | 22.97 | 15.44 | 19.25 |
| CNN  | 19.11 | 22.81 | 17.58 | 19.90 |
| LSTM | 15.36 | 20.16 | 15.39 | 17.43 |

### MAE + 헤드별 (Transformer 인코더)

| 모델 | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|------|-----------|-----------|-----------|-----------|
| MAE + MLP  | **14.86** | **18.77** | **14.31** | **15.94** |
| MAE + CNN  | 15.27 | 18.81 | 14.90 | 17.00 |
| MAE + LSTM | 52.89 | 26.35 | 43.50 | 19.44 |

## 평가 지표

- **RMSE**: 예측값과 실제 RUL 간의 평균 제곱근 오차
- **NASA Score**: 고장 임박 구간의 늦은 예측에 비대칭 패널티를 부여하는 도메인 특화 지표

## 참고

- 데이터: [NASA C-MAPSS](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- 논문: Saxena et al. (2008), PHM08
