import os
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput

# GPU設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset_path = "output/output85/output85_train.pt"
dev_dataset_path = "output/output85/output85_dev.pt"

# データセットの読み込み
train_dataset = torch.load(train_dataset_path)
dev_dataset = torch.load(dev_dataset_path)

# label→labelsに変更，textもpopで削除（BatchEncodingを使用していたらこんなことにはならなかった．）
for dataset in [train_dataset, dev_dataset]:
    for data in dataset:
        data["labels"] = torch.tensor(data.pop("label"), dtype=torch.long)
        if "text" in data:
            data.pop("text")

# トークナイザのロード
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

# DataCollatorWithPaddingの準備
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding='longest', # バッチ内で最長のものに合わせてパ
    return_tensors='pt'
)


# Attention Poolingを用いた分類モデルの定義
class BertAttentionPoolClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # トークンごとの重要度を計算する簡単な線形層
        self.attention = nn.Linear(self.hidden_size, 1)
        
        # 分類器
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # BERTの出力
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # PAD部分はマスク
        mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        hidden = hidden * mask  # PAD部分は0に

        # Attentionスコアの計算
        attn_scores = self.attention(hidden).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # 文表現の計算（加重平均）
        pooled = torch.sum(hidden * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden)

        logits = self.classifier(pooled)

        # loss計算
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

# モデルのロード（分類用に調整）
# clsトークンを内部的に使用している．
model = BertAttentionPoolClassifier(
    model_name="google-bert/bert-base-uncased",
    num_labels=2
).to(device)

training_args = TrainingArguments(
    output_dir="output/output89", # 出力ディレクトリ
    per_device_train_batch_size=64, # 学習用:1デバイスあたりのバッチサイズ
    per_device_eval_batch_size=64, # 評価用:1デバイスあたりのバッチサイズ
    learning_rate=2e-5, # 学習率
    lr_scheduler_type="linear", # 学習率スケジューラの種類
    warmup_steps=500, # ウォームアップステップ数
    num_train_epochs=6, # エポック数
    save_strategy="epoch", # モデル保存のタイミング
    logging_steps=100, # ロギングの頻度
    eval_strategy="epoch", # 評価のタイミング
    load_best_model_at_end=True, # 最良モデルのロード
    metric_for_best_model="accuracy", # 最良モデルの指標
    fp16=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=1)

    labels = labels.reshape(-1)

    return {
        "accuracy": (pred == labels).astype(np.float32).mean().item()
    }


trainer = Trainer(
    model=model,
    eval_dataset=dev_dataset,
    train_dataset=train_dataset,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics
)

trainer.train()

# --- ここに差し込む！ ---
print("=== 検証セットの予測サンプル確認 ===")
predictions = trainer.predict(dev_dataset)
# 生の出力（Logits）を確認
print("Logits (first 10):")
print(predictions.predictions[:10]) 

# クラス確定（0か1か）を確認
predicted_classes = np.argmax(predictions.predictions, axis=1)
print("Predicted Classes (first 10):")
print(predicted_classes[:10])

# 正解ラベルとの比較
true_labels = predictions.label_ids
print("True Labels (first 10):")
print(true_labels[:10])

# 検証セットでモデルを評価
eval_metrics = trainer.evaluate()
os.makedirs("output", exist_ok=True)
output_file = "output/output89.txt"
with open(output_file, "w", encoding="utf-8") as output_f:
    for key, value in eval_metrics.items():
        output_f.write(f"{key}: {value}\n")