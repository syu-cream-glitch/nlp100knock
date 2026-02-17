import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


# データ読み込み
def load_sst2_data(filepath):
    """
    SST-2データを読み込む
    tsvファイルの1列目がsentence, 2列目がlabel
    """
    df = pd.read_csv(filepath, sep='\t', header=0)
    df = df.dropna(subset=["sentence", "label"])
    sentences = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return sentences, labels


# GRPO用データセット作成
def create_grpo_dataset(sentences, labels):
    data = [{"prompt": f"Classify the sentiment of the following sentence.\nSentence: {s}\nAnswer:", "label": l}
            for s, l in zip(sentences, labels)]
    return Dataset.from_list(data)


# 報酬関数
def reward_func(prompts, generations):
    rewards = []
    for prompt, gen in zip(prompts, generations):
        # prompt からラベルを取得
        if "Sentence:" in prompt:
            sentence = prompt.split("Sentence:")[1].split("\nAnswer:")[0].strip()
            label = train_label_dict.get(sentence, 0)  # train_label_dict: {sentence: label}
        else:
            label = 0

        text = gen.lower()
        if label == 1 and "positive" in text:
            rewards.append(1.0)
        elif label == 0 and "negative" in text:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# モデルの評価関数
def evaluate_model(model, tokenizer, sentences, labels, device, max_new_tokens=5):
    model.eval()
    correct = 0
    total = len(sentences)
    for sentence, true_label in zip(sentences, labels):
        prompt = f"Classify the sentiment of the following sentence.\nSentence: {sentence}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
        if "positive" in answer:
            pred_label = 1
        elif "negative" in answer:
            pred_label = 0
        else:
            pred_label = 0
        if pred_label == true_label:
            correct += 1
    return correct / total


# GPU設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# モデル・トークナイザ
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map={"": 0}
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id


# データロード
train_sents, train_labels = load_sst2_data("SST-2/train.tsv")
dev_sents, dev_labels     = load_sst2_data("SST-2/dev.tsv")

train_sents = train_sents[:1000]
train_labels = train_labels[:1000]

train_label_dict = dict(zip(train_sents, train_labels))

train_dataset = create_grpo_dataset(train_sents, train_labels)
eval_dataset  = create_grpo_dataset(dev_sents, dev_labels)


# GRPO設定
grpo_config = GRPOConfig(
    output_dir="output/output99",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-7,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,
    reward_func=reward_func,
    num_return_sequences=4,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# GRPO学習前のモデル精度を確認
accuracy_before = evaluate_model(model, tokenizer, dev_sents[:100], dev_labels[:100], device)
print(f"GRPO適用前の精度: {accuracy_before:.4f}（最初の100件での評価）")

# GRPOによる選好学習の実行
trainer.train()

# GRPO学習後のモデル精度を確認
accuracy_after = evaluate_model(model, tokenizer, dev_sents[:100], dev_labels[:100], device)
print(f"GRPO適用後の精度: {accuracy_after:.4f}（最初の100件での評価）")
print(f"精度の向上幅: {accuracy_after - accuracy_before:.4f}")

# 評価結果
output_file = os.path.join("output", "output99.txt")
os.makedirs("output", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"学習データ件数: {len(train_sents)} 件\n")
    output_f.write(f"評価データ件数: {len(dev_sents)} 件\n")
    output_f.write(f"GRPO適用前精度: {accuracy_before:.4f}\n")
    output_f.write(f"GRPO適用後精度: {accuracy_after:.4f}\n")
    output_f.write(f"精度の向上量: {accuracy_after - accuracy_before:.4f}\n")
