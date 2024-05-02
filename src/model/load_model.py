from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
from src.features.load_images import LoadImages


class GetSimilarityFeatures():
    """
    Classe pour calculer et enregistrer les caractéristiques de similarité entre les images.
    """
    def __init__(self, model_path):
        self.model_path = model_path
    
    def calculate_similarity(self, images, game_db):
        """
        Calcule la similarité entre les images et enregistre les résultats dans un DataFrame.

        Parameters:
        - images (numpy.ndarray): Tableau numpy contenant les features des images.
        - game_db (DataFrame): Base de données des jeux.
        """
        feat_extractor = pickle.load(open(self.model_path, 'rb'))
        imgs_features = feat_extractor.predict(images)
        # Calculer la cosine similarité entre les images
        cosSimilarities = cosine_similarity(imgs_features)
        # Enregistrer les résultats dans un dataframe
        cos_similarities_df = pd.DataFrame(cosSimilarities, columns=game_db.index, index=game_db.index)
        cos_similarities_df.to_parquet("data/metrics/cos_sim_features_5k.parquet")


# Charger les données
imgs_model_width, imgs_model_height = 224, 224
game_db = pd.read_parquet("data/processed/game_database_5k.parquet")
image_paths = game_db['header_image'].tolist()
loader = LoadImages(image_paths)
images = loader.load_images(imgs_model_height, imgs_model_width)

# Récupérer les similarités des features
sim_features = GetSimilarityFeatures("models/vgg_model.pkl")
sim_features.calculate_similarity(images, game_db)