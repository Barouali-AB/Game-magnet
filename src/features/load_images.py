import requests
from io import BytesIO
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

class LoadImages:
    """
    Classe pour charger et prétraiter les images à partir de chemins d'accès donnés.
    """
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def load_images(self, img_height, img_width):
        """
        Charge et prétraite les images selon les dimensions spécifiées.

        Parameters:
        - img_height (int): Hauteur des images après redimensionnement.
        - img_width (int): Largeur des images après redimensionnement.

        Returns:
        numpy.ndarray: Tableau numpy contenant les images prétraitées.
        """
        images = []

        for path in self.image_paths:
            # Récupérer le contenu de l'image
            response = requests.get(path)
            # Ouvrir l'image à partir du contenu de la réponse
            img = Image.open(BytesIO(response.content))
            # Redimensionner l'image
            img = img.resize((img_height, img_width))
            # Convertir en tableau numpy et normaliser
            img_array = img_to_array(img) / 255.0 
            # Ajouter une dimmension supplémentaire => 4 dimensions 
            images = np.expand_dims(img_array, axis=0)
        # Empiler les images dans un seul tableau
        images = np.vstack(images)
        # Pré-traiter les images pour les adapter au modèle
        processed_imgs = preprocess_input(images.copy())

        return processed_imgs