import os
import logging
import pandas as pd
from src.model import train_model  # 기존 모델 학습 함수
from src.pretrained_model import fine_tune_model  # 사전학습 모델 fine-tuning 함수
import joblib
import boto3
import io
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)

MODEL_PERFORMANCE_THRESHOLD: float = 0.8  # 검증 정확도 기준
DATA_SHIFT_P_VALUE_THRESHOLD: float = 0.05  # KS 검정 p-value 기준


def download_data_from_s3(bucket_name: str, key: str) -> pd.DataFrame:
    """
    AWS S3에서 CSV 파일을 다운로드하여 DataFrame으로 반환합니다.
    """
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=key)
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    return df


def evaluate_existing_model(model: any, X_val: pd.Series, y_val: pd.Series) -> float:
    predictions = model.predict(X_val)
    return accuracy_score(y_val, predictions)


def detect_data_shift(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_col: str = 'query_clean') -> bool:
    train_lengths = train_df[feature_col].apply(lambda x: len(x.split()))
    val_lengths = val_df[feature_col].apply(lambda x: len(x.split()))
    statistic, p_value = ks_2samp(train_lengths, val_lengths)
    logging.info(f"KS test statistic: {statistic:.4f}, p-value: {p_value:.4f}")
    return p_value < DATA_SHIFT_P_VALUE_THRESHOLD


def auto_retrain() -> None:
    """
    클라우드(S3) 기반 자동 재학습 스크립트:
      1. S3에서 학습 데이터와 검증 데이터를 다운로드
      2. 전처리 후 기존 모델 또는 사전학습 모델의 성능 평가
      3. 데이터 분포 shift 감지 (KS 검정)
      4. 모델 성능이 기준 미달이거나 데이터 shift가 감지되면 재학습 수행
         - 'baseline' 또는 'pretrained' 모드를 if/elif/else로 분기합니다.
    """
    bucket_name = os.getenv('DATA_BUCKET_NAME', 'my-data-bucket')
    training_data_key = os.getenv('TRAINING_DATA_KEY', 'data/query_intent_data.csv')
    validation_data_key = os.getenv('VALIDATION_DATA_KEY', 'data/validation_data.csv')

    try:
        logging.info("S3에서 학습 데이터 다운로드 중...")
        df_train = download_data_from_s3(bucket_name, training_data_key)
    except Exception as e:
        logging.error(f"학습 데이터 다운로드 실패: {e}")
        return

    try:
        logging.info("S3에서 검증 데이터 다운로드 중...")
        df_val = download_data_from_s3(bucket_name, validation_data_key)
    except Exception as e:
        logging.error(f"검증 데이터 다운로드 실패: {e}")
        return

    from src.preprocessing import clean_text
    df_train['query_clean'] = df_train['query'].apply(clean_text)
    df_val['query_clean'] = df_val['query'].apply(clean_text)

    # 기존 모델 평가 (모델 파일이 존재하면)
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
        X_val = df_val['query_clean']
        y_val = df_val['intent']
        accuracy = evaluate_existing_model(model, X_val, y_val)
        logging.info(f"기존 모델 검증 정확도: {accuracy:.4f}")
    else:
        logging.info("기존 모델 파일이 없으므로 재학습 진행합니다.")
        accuracy = 0.0

    data_shift = detect_data_shift(df_train, df_val, feature_col='query_clean')

    # 재학습 조건 판단
    if accuracy < MODEL_PERFORMANCE_THRESHOLD or data_shift:
        logging.info("모델 성능 저하 또는 데이터 분포 변화 감지됨 - 재학습 진행합니다.")
        retrain_mode: str = os.getenv('RETRAIN_MODE', 'pretrained')  # 'baseline' 또는 'pretrained'
        if retrain_mode == 'pretrained':
            logging.info("Transformer 기반 사전학습 모델 재학습 시작")
            fine_tune_model('data/query_intent_data.csv', output_dir='./pretrained_model', epochs=3)
        elif retrain_mode == 'baseline':
            logging.info("기존 모델 재학습 시작")
            train_model('data/query_intent_data.csv')
        else:
            logging.error(f"알 수 없는 retrain_mode: {retrain_mode}. 기본값(pretrained)으로 진행합니다.")
            fine_tune_model('data/query_intent_data.csv', output_dir='./pretrained_model', epochs=3)
        logging.info("모델 재학습 및 업데이트 완료!")
    else:
        logging.info("모델 성능 및 데이터 분포가 양호하여 재학습 불필요합니다.")


if __name__ == '__main__':
    auto_retrain()
