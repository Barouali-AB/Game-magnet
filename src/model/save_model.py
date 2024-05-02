from keras.applications import vgg16
from keras.models import Model
import pickle

class VGGFeatureExtractor:
    """
    Classe pour charger et sauvegarder le modèle VGG16 pré-entraîné et extraire les caractéristiques des images.
    """
    def __init__(self):
        self.vgg_model = None
        self.feat_extractor = None
    
    def load_model(self):
        """
        Charge le modèle VGG16 pré-entraîné et configure l'extracteur de features.
        """
        # Charger le modèle VGG16
        self.vgg_model = vgg16.VGG16(weights='imagenet')
        # Créer un extracteur de features
        self.feat_extractor = Model(inputs=self.vgg_model.input, outputs=self.vgg_model.get_layer("fc2").output)
    
    def save_model(self, file_path):
        """
        Sauvegarde l'extracteur de features dans un fichier spécifié.

        Params:
        - file_path (str): Chemin d'accès du fichier de sauvegarde.
        """
        if self.feat_extractor is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(self.feat_extractor, f)
        else:
            raise ValueError("Le modèle n'est pas encore chargé.")
    
# Créer une instance de VGGFeatureExtractor
vgg_extractor = VGGFeatureExtractor()

# Charger le modèle
vgg_extractor.load_model()

# Enregistrer le modèle
vgg_extractor.save_model('models/vgg_model.pkl')