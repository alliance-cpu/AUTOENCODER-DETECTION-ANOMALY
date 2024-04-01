import tkinter.filedialog as filedialog
import tkinter as tk
from data_preprocessing import preprocess_data
from autoencoder_model import AnomalyDetector
from visualization import *

def load_data():
    # Ouvrir une boîte de dialogue pour sélectionner le fichier de données
    file_path = filedialog.askopenfilename(filetypes=[("Fichiers CSV", "*.csv")])

    if file_path:
        data_normalized, autoencoder = preprocess_data(file_path)
        setup_visualization(data_normalized, autoencoder)
    else:
        # Afficher un message d'erreur si aucun fichier n'a été sélectionné
        message_label.config(text="Aucun fichier sélectionné.")
