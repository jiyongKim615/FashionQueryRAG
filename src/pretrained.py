# src/pretrained_model.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Any


class QueryDataset(Dataset):
    def __init__(self, encodings: Dict[str, Any], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def load_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def encode_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 128) -> Tuple[
    QueryDataset, Dict[str, int]]:
    texts = df['query'].tolist()
    labels = df['intent'].tolist()
    label_set = list(set(labels))
    label2id = {label: idx for idx, label in enumerate(label_set)}
    df['label'] = df['intent'].map(label2id)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    dataset = QueryDataset(encodings, df['label'].tolist())
    return dataset, label2id


def fine_tune_model(data_path: str, model_name: str = 'bert-base-multilingual-cased',
                    output_dir: str = './pretrained_model', epochs: int = 3) -> Tuple[
    Any, AutoTokenizer, Dict[str, int]]:
    df = load_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset, label2id = encode_dataset(df, tokenizer)
    # 데이터셋 분할 (80% train, 20% eval)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer, label2id


def predict_pretrained(query: str, model_dir: str = './pretrained_model',
                       label2id: Dict[str, int] | None = None) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    if label2id:
        id2label = {v: k for k, v in label2id.items()}
        return id2label.get(pred, str(pred))
    return str(pred)


if __name__ == '__main__':
    model, tokenizer, label2id = fine_tune_model('data/query_intent_data.csv')
    test_query = "여성 봄 코트 추천"
    prediction = predict_pretrained(test_query, label2id=label2id)
    print("Pretrained model prediction:", prediction)
