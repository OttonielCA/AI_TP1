import tensorflow as tf # Importe la bibliothèque TensorFlow
import numpy as np # Importe la bibliothèque NumPy
from matplotlib import pyplot as plt # Importe la bibliothèque Matplotlib pour la visualisation

def show_mnist_graphic_number(img): # Définit une fonction pour afficher une image MNIST
    img = np.array(img, dtype='float') # Convertit l'image en tableau NumPy de type float
    pixels = img.reshape((28, 28)) # Convertit l'image en un tableau 2D de 28x28 pixels
    plt.imshow(pixels, cmap='gray') # Affiche l'image en niveaux de gris
    plt.show() # Affiche l'image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Charge les données MNIST

new_model = tf.keras.models.load_model('tp1.keras') # Charge le modèle enregistré

new_model.summary() # Affiche un résumé de l'architecture du modèle

# Affiche les prédictions demandées par Steve ;)
img = [1, 6, 3513, 145, 458] # Définit les indices des images à prédire
for x in img: # Parcourt les indices
    img1 = x_test[x] # Récupère l'image correspondante
    print(new_model.predict(np.reshape(img1, (1, 28, 28))))
    show_mnist_graphic_number(img1) # Affiche l'image correspondante
