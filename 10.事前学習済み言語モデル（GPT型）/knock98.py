import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from tqdm import tqdm

# GPU設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_files = [
    "SST-2/train.tsv",
    "SST-2/dev.tsv"
]

# データの読み込み
dfs = [pd.read_csv(f, sep='\t') for f in input_files]
train_df, dev_df = dfs

# あまりにも長いため、学習データを20000件に制限
train_df = train_df.iloc[:20000]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map={"": 0}
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

class SST2GenDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["sentence"]
        label = "positive" if row["label"] == 1 else "negative"

        prompt = f"""Classify the sentiment of the following sentence.
Sentence: {text}
Answer:"""

        # プロンプトを部分を今後除外するためにプロンプト部分だけをトークナイズしておく
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        full_text = prompt + " " + label

        enc = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        labels = input_ids.clone()

        # プロンプト部分を-100に設定（学習対象外）→内部的にignore_index=-100とされているから-100じゃないといけない．
        prompt_len = len(prompt_enc["input_ids"][0])
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

train_dataset = SST2GenDataset(train_df, tokenizer)
dev_dataset   = SST2GenDataset(dev_df, tokenizer)

training_args = TrainingArguments(
    output_dir="output/output98", # 出力ディレクトリ
    per_device_train_batch_size=64, # 学習用:1デバイスあたりのバッチサイズ
    per_device_eval_batch_size=64, # 評価用:1デバイスあたりのバッチサイズ
    learning_rate=2e-5, # 学習率
    lr_scheduler_type="linear", # 学習率スケジューラの種類
    warmup_steps=10, # データ数が少ないためウォームアップも減らす
    num_train_epochs=1, # エポック数
    save_strategy="epoch", # モデル保存のタイミング
    logging_steps=5, # ロギングの頻度
    eval_strategy="epoch", # 評価のタイミング
    load_best_model_at_end=True, # 最良モデルのロード
    metric_for_best_model="eval_loss",
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

trainer.train()

# --- dev データで生成→accuracy計算 ---
model.eval() # 評価モード
correct = 0
total = len(dev_df)

for i in tqdm(range(total)):
    row = dev_df.iloc[i]
    text = row["sentence"]
    true_label = "positive" if row["label"] == 1 else "negative"

    # 推論時は「正解ラベルを含まない」プロンプトのみを作成
    prompt = f"""Classify the sentiment of the following sentence.
Sentence: {text}
Answer:"""

    # プロンプトのみをトークナイズ
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=5, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # プロンプト部分をスキップして生成部分だけを取得
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

    if true_label in answer:
        correct += 1

accuracy = correct / total
print(f"Accuracy on dev set: {accuracy:.3f}")

# 出力ファイルに保存
output_file = os.path.join("output", "output98.txt")
os.makedirs("output", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"Accuracy on dev set: {accuracy:.3f}\n")