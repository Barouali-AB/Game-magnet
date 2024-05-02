import pandas as pd
import numpy as np
from angle_emb import AnglE
from io import BytesIO
from PIL import Image
import numpy as np
import requests


# Charger les données
df_games = pd.read_csv('data/raw/steam.csv')
df_media = pd.read_csv('data/raw/steam_media_data.csv')
df_desc = pd.read_csv('data/raw/steam_description_data.csv')
df = df_games.merge(df_media, left_on='appid', right_on='steam_appid').merge(df_desc, on='steam_appid')

# Utiliser l'embedding AnglE
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

def is_colored(image_url):
    """
    Retourne si une image est colorée ou pas selon son URL
    
    Params:
        - image_url (str): URL de l'image.
    
    Returns:
    bool : True ou False si l'image est en couleur ou pas
  
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img_array = np.array(img)
    return len(img_array.shape) == 3 and img_array.shape[2] == 3

# Récupérer les jeux avec le plus grand nombre de owners
df['minimum_owners'] = df['owners'].apply(lambda v : int(v.split('-')[0]) )
top_df = df.sort_values('minimum_owners', ascending=False)[:5000]

# Filtrer sur les images colorées 
top_df['is_colored'] = top_df['header_image'].apply(is_colored)
top_df = top_df[top_df['is_colored'] == True]

# Créer une nouvelle colonne contenant nom, developpeur, genre et description
game_texts = []
for _, row in top_df.iterrows():
  game_text = f'''Name: {row['name']}
Developer: {row['developer']}
Genres: {row['genres']}
Summary: {row['short_description']}'''
  game_texts.append(game_text)

top_df['text'] = game_texts
print(len(top_df))

# Calcule des embeddings à partir de la colonne text
embeddings = []
batches = np.array_split(top_df['text'], len(top_df) // 5)
for idx, chunk_text in enumerate(batches):
  if idx % 5 == 0 :
    print(f"{idx} / {len(batches)}")
  embeddings += list(angle.encode(list(chunk_text), to_numpy=True))

top_df['embedding'] = embeddings
print(len(top_df))

# Supprimer les colonnes inutiles
top_df.drop(columns = ['release_date', 'english', 'publisher',
       'platforms', 'required_age', 'categories', 'steamspy_tags',
       'achievements', 'positive_ratings', 'negative_ratings',
       'average_playtime', 'median_playtime', 'price', 'screenshots', 'background', 'movies',
       'detailed_description', 'about_the_game', 'owners'], inplace=True)

top_df.reset_index(drop=True, inplace=True)
print(len(top_df))

# Enregistrer le dataset
top_df.to_parquet('data/processed/game_database_5k.parquet')