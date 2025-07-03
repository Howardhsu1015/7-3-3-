# 7-3-3-
# 安裝必要套件
!pip install -q sentence-transformers pandas scikit-learn

# 匯入函式庫
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 讀取歌曲資料集
songs = pd.read_csv("/mnt/data/song_dataset.csv")

# 使用預訓練模型將歌詞轉換為向量
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(songs['lyrics'].tolist())

# 建立推薦函數
def recommend_songs(user_input, top_n=5):
    input_embedding = model.encode([user_input])
    similarities = cosine_similarity(input_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    return songs.iloc[top_indices][['title', 'artist', 'genre', 'lyrics']]

# 使用者輸入（你可以改成任何歌詞片段或風格描述）
user_text = "我喜歡抒情又深刻的任何台灣小眾樂團的音樂歌詞"

# 顯示推薦結果
recommendations = recommend_songs(user_text)
print("推薦歌曲：\n")
print(recommendations)
