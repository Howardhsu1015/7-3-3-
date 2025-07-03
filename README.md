import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    # 讀取歌曲歌詞資料（CSV檔）
    df = pd.read_csv('songs_lyrics.csv')
    texts = df['lyrics'].fillna('').tolist()[:50]  # 取前50筆歌詞

    # 載入句子嵌入模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 產生歌詞向量嵌入
    embeddings = model.encode(texts, show_progress_bar=True)

    # 轉成 numpy 陣列並儲存
    embeddings = np.array(embeddings)
    np.save('lyrics_embeddings.npy', embeddings)

    print(f"成功產生並儲存 {embeddings.shape[0]} 筆 embedding，維度為 {embeddings.shape[1]}")

if __name__ == '__main__':
    main()
