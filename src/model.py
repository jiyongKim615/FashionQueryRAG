# src/model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
from src.preprocessing import clean_text
from typing import Tuple, Union

def load_data(data: Union[str, pd.DataFrame]) -> Tuple[pd.Series, pd.Series]:
    """
    data: 파일 경로나 DataFrame.
    query 컬럼에 대해 간단한 전처리(clean_text)를 수행합니다.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data
    df['query_clean'] = df['query'].apply(clean_text)
    return df['query_clean'], df['intent']

def train_model(data: Union[str, pd.DataFrame]) -> Pipeline:
    X, y = load_data(data)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, 'model.pkl')
    return pipeline

def predict(query: str, model_path: str = 'model.pkl') -> str:
    pipeline = joblib.load(model_path)
    query_clean = clean_text(query)
    return pipeline.predict([query_clean])[0]

if __name__ == '__main__':
    model = train_model('data/query_intent_data.csv')
    test_query = "남성 정장 셔츠 추천 좀 해줘"
    result = predict(test_query)
    print(f"'{test_query}' 의 검색 의도: {result}")
