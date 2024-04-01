import tkinter as tk
from anomaly_detection import load_data

# Créer la fenêtre principale
window = tk.Tk()
window.title("Détection d'anomalies dans un réseau informatique")

# Ajouter un bouton pour charger les données
load_button = tk.Button(window, text="Charger les données", command=load_data)
load_button.pack()

# Ajouter une étiquette pour afficher un message de confirmation ou d'erreur
message_label = tk.Label(window, text="")
message_label.pack()

# Exécuter la boucle principale de l'interface graphique
window.mainloop()
