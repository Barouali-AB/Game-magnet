import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from fuzzywuzzy import process
import time 

class GameRecommendation:
    def __init__(self, game_db, cos_sim, k=10):
        self.game_db = game_db
        self.cos_sim = cos_sim
        self.k = k

    def game_finder(self, name):
        """
        Trouve le nom de jeu le plus proche correspondant à l'entrée fournie.

        Params :
        - name (str) : Le nom du jeu en entrée.

        Returns :
        str : Le nom du jeu le plus proche trouvé dans la base de données.
        """
        all_names = self.game_db['name'].tolist()
        closest_match = process.extractOne(name, all_names)
        return closest_match[0]

    def get_recommendation(self, game_name):
        """
        Obtient des recommandations de jeux similaires en fonction du nom du jeu fourni en se basant sur les embeddings des textes.

        Params:
        - game_name (str): Le nom du jeu pour lequel obtenir des recommandations.

        Returns:
        tuple: Temps d'exécution en secondes (float), et une liste des noms de jeux similaires (list[str]).
        """
        game_name = self.game_finder(game_name)
        game_row = self.game_db[self.game_db['name'] == game_name]
        start_time = int(round(time.time() * 1000000))
        distances = cosine_distances(np.array(game_row.embedding.tolist()), np.array(self.game_db.embedding.tolist()))[0]
        sorted_indices = distances.argsort()
        end_time = int(round(time.time() * 1000000))
        similar_games = []
        for idx in sorted_indices[1:int(self.k)+1].tolist():
            similar_games.append(self.game_db['name'].iloc[idx])
        return end_time - start_time, similar_games
    
    def get_recommendation_img(self, game_name):
        """
        Obtient des recommandations de jeux similaires en fonction du nom du jeu fourni en se basant sur la matrice de similarité 
        entre les features des images.

        Params:
        - game_name (str): Le nom du jeu pour lequel obtenir des recommandations.

        Returns:
        dict: Un dictionnaire contenant les noms de jeux similaires en tant que clés (str) et leurs images en tant que valeurs (str).
        """
        game_name = self.game_finder(game_name)
        game_row = self.game_db[self.game_db['name'] == game_name]
        closest_imgs = self.cos_sim.loc[game_row.index[0]].argsort()[::-1][1:self.k+1].tolist()
        similar_games = {}
        for i in range(self.k):
            similar_games[self.game_db['name'].iloc[closest_imgs[i]]] = self.game_db['header_image'].iloc[closest_imgs[i]]
        return similar_games