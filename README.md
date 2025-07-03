**import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path='lyrics_embeddings.npy'):
    return np.load(path)

def load_data(path='songs_lyrics.csv'):
    return pd.read_csv(path)

def find_top_k_similar(query_embedding, all_embeddings, k=5):
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    # 排序索引，從高到低
    sorted_indices = similarities.argsort()[::-1]
    # 移除自己（相似度最高的自己）
    top_k_indices = [i for i in sorted_indices if similarities[i] < 0.9999][:k]
    top_k_scores = similarities[top_k_indices]
    return top_k_indices, top_k_scores

def main():
    df = load_data()
    embeddings = load_embeddings()

    # 以第0筆資料為例，找最相似的前5首歌
    query_emb = embeddings[0]
    indices, scores = find_top_k_similar(query_emb, embeddings, k=5)

    print(f"與歌曲 '{df.iloc[0]['title']}' 相似的前5首歌曲:")
    for idx, score in zip(indices, scores):
        print(f"{df.iloc[idx]['title']} - 相似度: {score:.4f}")

if __name__ == "__main__":
    main()
**
