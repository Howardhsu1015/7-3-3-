import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data(csv_path):
    """
    載入CSV資料，預期欄位：title, artist, genre, lyrics
    """
    df = pd.read_csv(csv_path)
    # 確保歌詞欄位沒有缺失
    df['lyrics'] = df['lyrics'].fillna('')
    return df

def build_tfidf_matrix(df):
    """
    建立歌詞的TF-IDF向量矩陣
    """
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['lyrics'])
    return tfidf_matrix

def compute_similarity(tfidf_matrix):
    """
    計算餘弦相似度矩陣
    """
    return linear_kernel(tfidf_matrix, tfidf_matrix)

def make_indices(df):
    """
    建立標題->索引的映射
    """
    return pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend_songs(title, df, cosine_sim, indices, top_n=5):
    """
    根據輸入歌曲標題，推薦相似歌曲
    """
    if title not in indices:
        return f"歌曲 '{title}' 不在資料中。"
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # 排除自己

    song_indices = [i[0] for i in sim_scores]
    return df.iloc[song_indices][['title', 'artist', 'genre']].reset_index(drop=True)

def main():
    # 你要測試的CSV檔路徑
    csv_path = 'songs_lyrics.csv'

    # 載入資料
    df = load_data(csv_path)

    # 建立向量及計算相似度
    tfidf_matrix = build_tfidf_matrix(df)
    cosine_sim = compute_similarity(tfidf_matrix)
    indices = make_indices(df)

    # 範例：輸入一首歌推薦相似歌曲
    test_title = 'Freedom'  # 請確保CSV裡有這首歌
    print(f"推薦與 '{test_title}' 相似的歌曲：")
    recommendations = recommend_songs(test_title, df, cosine_sim, indices, top_n=5)
    print(recommendations)

if __name__ == '__main__':
    main()
