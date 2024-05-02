from annoy import AnnoyIndex
from fuzzywuzzy import process
import time
import time

    
class ApproximateNearestNeighbors:
    """
    Classe pour effectuer des recherches d'approximation des voisins les plus proches.
    """
    def __init__(self, game_db, num_trees=10, k = 10):
        self.num_trees = num_trees
        self.game_db = game_db
        self.k = k
        self.index = None

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

    def build_index(self, embeddings):
        """
        Construit l'index à partir des embeddings donnés.

        Params:
        - embeddings (list): La liste des embeddings des jeux.
        """
        num_dimensions = len(embeddings[0])
        self.index = AnnoyIndex(num_dimensions, 'angular')  
        for i, embedding in enumerate(embeddings):
            self.index.add_item(i, embedding)
        self.index.build(self.num_trees)

    def get_recommendations(self, game_name):
        """
        Obtient des recommandations de jeux similaires en se basant sur l'index construit.

        Parameters:
        - game_name (str): Le nom du jeu d'entrée.

        Returns:
        tuple: Temps d'exécution en secondes (float), et une liste des noms de jeux similaires (list[str]).
        """
        if self.index is None:
            raise Exception("Index pas encore construit. Faire un appel de build_index() avant.")
        game_name = self.game_finder(game_name)
        game_row = self.game_db[self.game_db['name'] == game_name]
        start_time = int(round(time.time() * 1000000))
        game_embedding = game_row['embedding'].tolist()[0]
        indices = self.index.get_nns_by_vector(game_embedding, self.k + 1)
        end_time = int(round(time.time() * 1000000))
        return end_time - start_time, [self.game_db['name'].iloc[idx] for idx in indices[1:]]