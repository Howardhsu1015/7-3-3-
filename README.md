import pandas as pd
import random

titles = [f"Song {i}" for i in range(1, 101)]
artists = [f"Artist {chr(65 + i%26)}" for i in range(100)]
genres = ['Pop', 'Rock', 'Classical', 'Indie', 'Folk', 'Blues']
lyrics_base = [
    "Love is in the air, shining bright tonight",
    "Dancing under the moonlight with you",
    "The mountains are calling, let's go explore",
    "Sunrise paints the sky with golden light",
    "Raindrops falling down, memories come back",
    "Feel the freedom running through your veins",
    "Whispers of the wind, secrets to unfold",
    "Hold my hand and never let me go",
    "Dreams are stars that guide us home",
    "Together we can face the darkest night"
]

data = []
for i in range(100):
    title = titles[i]
    artist = artists[i]
    genre = random.choice(genres)
    lyrics = " ".join(random.choices(lyrics_base, k=5))
    data.append([title, artist, genre, lyrics])

df = pd.DataFrame(data, columns=['title', 'artist', 'genre', 'lyrics'])
df.to_csv('songs_lyrics.csv', index=False)
print("songs_lyrics.csv 已生成，包含100筆資料")
