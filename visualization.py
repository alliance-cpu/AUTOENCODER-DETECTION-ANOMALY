import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np

def show_normal_ecg(data_normalized):
    plt.plot(data_normalized[0])
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    plt.title('ECG normal')
    plt.show()

def show_reconstruction(data_normalized, autoencoder):
    reconstruction = autoencoder.predict(data_normalized)
    plt.plot(reconstruction[0])
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    plt.title('Reconstruction')
    plt.show()

def show_reconstruction_error(data_normalized, autoencoder):
    reconstruction = autoencoder.predict(data_normalized)
    mse = np.mean(np.power(data_normalized - reconstruction, 2), axis=1)
    plt.plot(mse)
    plt.xlabel('Temps')
    plt.ylabel('Erreur')
    plt.title('Erreur de reconstruction')
    plt.show()

def show_combined(data_normalized, autoencoder):
    reconstruction = autoencoder.predict(data_normalized)
    mse = np.mean(np.power(data_normalized - reconstruction, 2), axis=1)
    plt.plot(reconstruction[0], label='Reconstruction')
    plt.plot(mse, label='Erreur de reconstruction')
    plt.xlabel('Temps')
    plt.ylabel('Amplitude / Erreur')
    plt.title('Reconstruction et Erreur de reconstruction')
    plt.legend()
    plt.show()

def show_histogram(data_normalized, autoencoder):
    reconstruction = autoencoder.predict(data_normalized)
    mse = np.mean(np.power(data_normalized - reconstruction, 2), axis=1)
    normal_samples = data_normalized[:100]
    normal_mse = mse[:100]
    plt.hist(normal_mse, bins=50)
    plt.xlabel("Erreur de reconstruction")
    plt.ylabel("Fréquence")
    plt.title("Histogramme de l'erreur de reconstruction pour les échantillons normaux")
    plt.show()

def setup_visualization(data_normalized, autoencoder):
    window = tk.Tk()
    normal_ecg_button = tk.Button(window, text="Afficher l'ECG normal", command=lambda: show_normal_ecg(data_normalized))
    normal_ecg_button.pack()

    reconstruction_button = tk.Button(window, text="Afficher la reconstruction", command=lambda: show_reconstruction(data_normalized, autoencoder))
    reconstruction_button.pack()

    reconstruction_error_button = tk.Button(window, text="Afficher l'erreur de reconstruction", command=lambda: show_reconstruction_error(data_normalized, autoencoder))
    reconstruction_error_button.pack()

    combined_button = tk.Button(window, text="Afficher la reconstruction et l'erreur de reconstruction", command=lambda: show_combined(data_normalized, autoencoder))
    combined_button.pack()

    histogram_button = tk.Button(window, text="Afficher l'histogramme de l'erreur de reconstruction", command=lambda: show_histogram(data_normalized, autoencoder))
    histogram_button.pack()

    window.mainloop()
