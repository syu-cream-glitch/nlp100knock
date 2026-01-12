import os
import numpy as np
from dotenv import load_dotenv
from gensim.models import KeyedVectors

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')

def load_pretrained_embeddings(path, max_vocab=None, pad_token='<PAD>'):
    # ID=0に割り当てるパディング用トークンを準備

    wv = KeyedVectors.load_word2vec_format(path, binary=True)

    # 埋め込み次元数を取得
    embedding_dim = wv.vector_size

    # 語彙リストの取得
    vocab = wv.index_to_key
    if max_vocab is not None:
        vocab = vocab[:max_vocab]

    vocab_size = len(vocab) + 1

    # 埋め込み行列の初期化（3000001行，300列のゼロ行列を作成）4bytesのfloat32型でメモリ確保
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)    

    # トークンとトークンIDの双方向の対応付け辞書を作成
    token2id = {pad_token: 0}
    id2token = {0: pad_token}

    # embedding_matrix[token2id[token]] == wv[token]が成り立つ
    for i, token in enumerate(vocab, start=1):
        embedding_matrix[i] = wv[token]
        token2id[token] = i
        id2token[i] = token
    
    return embedding_matrix, token2id, id2token

embedding_matrix, token2id, id2token = load_pretrained_embeddings(path, max_vocab=None)

os.makedirs('output', exist_ok=True)
output_file = os.path.join('output', 'output70.txt')
with open(output_file, 'w', encoding='utf-8') as output_f:
    output_f.write(f"語彙数: {len(token2id)}\n")
    output_f.write(f"次元数: {embedding_matrix.shape[1]}\n")
    output_f.write(f"形状: {embedding_matrix.shape}\n")
    output_f.write("最初の5単語\n")
    for i in range(1, 6):
        output_f.write(f"ID: {i}, 単語: {id2token[i]}, 埋め込みベクトル: {embedding_matrix[i]}\n")