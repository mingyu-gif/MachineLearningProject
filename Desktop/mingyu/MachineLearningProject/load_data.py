"""
NASA C-MAPSS 데이터 로드 및 RUL 레이블 생성
LibEER load_data.py 구조 참고하여 구현
"""
import numpy as np
import pandas as pd
import os

# 컬럼명 정의
COLUMNS = (
    ['unit', 'cycle'] +
    [f'os_{i}' for i in range(1, 4)] +
    [f's_{i}' for i in range(1, 22)]
)

# 정보없는 센서 (분산=0, 제거 대상)
DROP_SENSORS = ['s_1','s_5','s_6','s_10','s_16','s_18','s_19']

def load_cmapss(data_path, fd='FD001', max_rul=125):
    """
    C-MAPSS 데이터 로드 및 RUL 레이블 생성
    data_path: 데이터 폴더 경로
    fd       : 'FD001' ~ 'FD004'
    max_rul  : RUL 클리핑 최대값 (piece-wise linear RUL)
    반환:
      train_data : (N_train, n_features) numpy
      train_label: (N_train,) RUL 값
      test_data  : (N_test, n_features) numpy
      test_label : (N_test,) RUL 값
    """
    # 파일 로드
    train_path = os.path.join(data_path, f'train_{fd}.txt')
    test_path  = os.path.join(data_path, f'test_{fd}.txt')
    rul_path   = os.path.join(data_path, f'RUL_{fd}.txt')

    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLUMNS)
    test_df  = pd.read_csv(test_path,  sep=r'\s+', header=None, names=COLUMNS)
    rul_df   = pd.read_csv(rul_path,   sep=r'\s+', header=None, names=['RUL'])

    # RUL 생성 (train: 마지막 사이클 기준 역산)
    train_df = _add_rul(train_df, max_rul)

    # RUL 생성 (test: 제공된 RUL 파일 사용)
    test_df = _add_test_rul(test_df, rul_df, max_rul)

    # 불필요한 센서 제거
    train_df = train_df.drop(columns=DROP_SENSORS + ['unit', 'cycle'])
    test_df  = test_df.drop(columns=DROP_SENSORS + ['unit', 'cycle'])

    # 데이터 / 레이블 분리
    train_label = train_df.pop('RUL').values.astype(np.float32)
    test_label  = test_df.pop('RUL').values.astype(np.float32)

    train_data = train_df.values.astype(np.float32)
    test_data  = test_df.values.astype(np.float32)

    return train_data, train_label, test_data, test_label

def _add_rul(df, max_rul):
    """train 데이터에 RUL 컬럼 추가 (piece-wise linear)"""
    max_cycle = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    df = df.merge(max_cycle, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    df = df.drop(columns=['max_cycle'])
    return df

def _add_test_rul(df, rul_df, max_rul):
    """test 데이터에 RUL 컬럼 추가"""
    # 각 엔진의 마지막 사이클에만 RUL 할당
    last_cycle = df.groupby('unit')['cycle'].max().reset_index()
    last_cycle.columns = ['unit', 'max_cycle']
    rul_df['unit'] = range(1, len(rul_df) + 1)
    last_cycle = last_cycle.merge(rul_df, on='unit')
    df = df.merge(last_cycle, on='unit')
    df['RUL'] = df['RUL'] + (df['max_cycle'] - df['cycle'])
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    df = df.drop(columns=['max_cycle'])
    return df

def normalize(train_data, test_data):
    """
    Min-Max 정규화 (train 통계로 test 정규화)
    """
    min_val = train_data.min(axis=0)
    max_val = train_data.max(axis=0)
    denom   = (max_val - min_val) + 1e-8

    train_norm = (train_data - min_val) / denom
    test_norm  = (test_data  - min_val) / denom

    return train_norm, test_norm

if __name__ == '__main__':
    DATA_PATH = "/home/affctiv/Desktop/mingyu/CMAPSS/6. Turbofan Engine Degradation Simulation Data Set"
    
    for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
        tr_d, tr_l, te_d, te_l = load_cmapss(DATA_PATH, fd)
        print(f"{fd}: train {tr_d.shape}, test {te_d.shape}")