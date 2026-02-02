import os
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

model = AutoModel.from_pretrained(
    "google-bert/bert-base-uncased"
)

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

model.eval()
cls_embeddings = []

with torch.no_grad():
    for sentence in sentences:
        # 手順1：文字列→トークン列（wordpiece）
        # 手順2：[CLS]と[SEP]の付与
        # 手順3：トークン列→ID列
        # 手順4：ID列→モデル入力のテンソル変換
        inputs = tokenizer(sentence, return_tensors="pt")

        # model(inputs_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        # 上記のコードを以下のように省略できる．
        outputs = model(**inputs)

        # (batch_size, seq_length, hidden_size)→(batch_size, hidden_size)にしたい．
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # 形状を(hidden_size,)
        cls_embeddings.append(cls_embedding.squeeze(0))

os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output83.txt")
with open(output_file, "w", encoding="utf-8") as output_f:
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            cosine_sim = cosine_similarity(
                cls_embeddings[i],
                cls_embeddings[j],
                dim=0
            )
            output_f.write(
                f"sentence {i + 1} → sentence {j + 1}: {cosine_sim.item()}\n"
            )
