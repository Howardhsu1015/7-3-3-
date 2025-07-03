import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(csv_path='songs_lyrics.csv'):
    return pd.read_csv(csv_path)

def load_embeddings(embedding_path='lyrics_embeddings.npy'):
    return np.load(embedding_path)

def embed_text(text, model):
    return model.encode([text])[0]

def find_top_k_similar(query_emb, embeddings, k=5):
    sims = cosine_similarity([query_emb], embeddings)[0]
    sorted_idx = sims.argsort()[::-1]
    top_indices = [i for i in sorted_idx if sims[i] < 0.9999][:k]
    top_scores = sims[top_indices]
    return top_indices, top_scores

def main():
    print("載入資料與模型中，請稍候...")
    df = load_data()
    embeddings = load_embeddings()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("系統準備完成！請輸入歌詞或關鍵字來取得推薦 (輸入 'exit' 離開)：")

    while True:
        user_input = input("輸入：")
        if user_input.strip().lower() == 'exit':
            print("感謝使用，結束程式。")
            break
        if not user_input.strip():
            print("請輸入有效文字。")
            continue

        query_emb = embed_text(user_input, model)
        top_indices, top_scores = find_top_k_similar(query_emb, embeddings, k=5)

        print(f"推薦的相似歌曲：")
        for idx, score in zip(top_indices, top_scores):
            title = df.iloc[idx]['title']
            artist = df.iloc[idx]['artist']
            print(f" - {title} by {artist} (相似度: {score:.4f})")
        print("-" * 40)

if __name__ == "__main__":
    main()
