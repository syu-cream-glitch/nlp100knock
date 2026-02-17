import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import evaluate
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

# GPU設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_files = [
    "SST-2/train.tsv",
    "SST-2/dev.tsv"
]

# データの読み込み
dfs = [pd.read_csv(f, sep='\t') for f in input_files]
train_df, dev_df = dfs

# モデルのロード
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    num_labels=2,
    device_map={"": 0}
)

model.config.pad_token_id = tokenizer.eos_token_id

class SST2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.texts = df['sentence'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {
            "input_ids": enc['input_ids'].squeeze(0),
            "attention_mask": enc['attention_mask'].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

train_dataset = SST2Dataset(train_df, tokenizer)
dev_dataset   = SST2Dataset(dev_df, tokenizer)

# 評価関数
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="output/output97", # 出力ディレクトリ
    per_device_train_batch_size=128, # 学習用:1デバイスあたりのバッチサイズ
    per_device_eval_batch_size=128, # 評価用:1デバイスあたりのバッチサイズ
    learning_rate=2e-5, # 学習率
    lr_scheduler_type="linear", # 学習率スケジューラの種類
    warmup_steps=500, # ウォームアップステップ数
    num_train_epochs=1, # エポック数: こうせんとガニサス動かん
    save_strategy="epoch", # モデル保存のタイミング
    logging_steps=100, # ロギングの頻度
    eval_strategy="epoch", # 評価のタイミング
    load_best_model_at_end=True, # 最良モデルのロード
    metric_for_best_model="accuracy", # 最良モデルの指標
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate()

output_file = os.path.join("output", "output97.txt")
os.makedirs("output", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"Evaluation results: {eval_results}\n")