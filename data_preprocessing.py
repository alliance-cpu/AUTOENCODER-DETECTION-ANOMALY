import pandas as pd
from sklearn.preprocessing import StandardScaler
from autoencoder_model import AnomalyDetector

def preprocess_data(file_path):
    # Charger les données à partir du fichier CSV
    data = pd.read_csv(file_path)

    # Prétraitement des données
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    # Construction du modèle d'auto-encodeur
    autoencoder = AnomalyDetector(num_features=data.shape[1])

    # Entraînement du modèle
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data_normalized, data_normalized, epochs=10, batch_size=32)

    return data_normalized, autoencoder
