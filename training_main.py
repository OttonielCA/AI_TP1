import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Chargement du dataset MNIST pour l'entraînement et le test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Nettoyage des données/Normalisation de l'ensemble d'entraînement pour être entre 0 et 1
x_train_normalized = x_train / 255
x_test_normalized = x_test / 255
# print(x_train_normalized[2900]) # Exemple de sortie 2900 de l'ensemble d'entraînement

# Création du réseau de neurones profond
def create_model(my_learning_rate):
    # Initialisation du modèle
    model = tf.keras.models.Sequential()
    # Couche de "flattening"
    model.add(layers.Flatten(input_shape=(28, 28)))
    # Première couche cachée avec 256 neurones
    model.add(layers.Dense(units=256, activation='relu'))
    # Deuxième couche cachée avec 128 neurones
    model.add(layers.Dense(units=128, activation='relu'))
    # Couche de dropout avec un taux de 0.2
    model.add(layers.Dropout(rate=0.2))
    # Couche de sortie
    model.add(layers.Dense(units=10, activation='softmax'))
    # Compilation du modèle
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model

# Entraînement du réseau de neurones profond
def train_model(model, train_features, train_label, epochs, batch_size=None, validation_split=0.1):
    # Entraînement du modèle
    history = model.fit(x=train_features, y=train_label, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=validation_split)
    # Suivi de la progression de l'entraînement
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

def plot_curve(epochs, hist, list_of_metrics): #Graphique combiné pour une meilleure lecture.
    plt.xlabel("Epochs")
    plt.ylabel("Valeur")
    plt.title("Métriques d'entraînement et de validation")  # Le titre de notre graphique

    for m in list_of_metrics:
        # Affichage de l'exactitude de 0 à 100% (meilleure visibilité)
        if 'accuracy' in m or 'val_accuracy' in m:
            x = hist[m] * 100
        else:
            x = hist[m]

        # Tracement du graphique
        plt.plot(epochs, x, label=m)

    plt.legend(loc='best')  # Placecement de notre légende
    plt.grid(True)  # Ajoute des lignes de grille pour une meilleure lisibilité
    plt.savefig('metrics_plot.png')  # Enregistre le graphique "metrics_plot.png"
    plt.show()

learning_rate = 0.01
epochs = 25
batch_size = 4000
validation_split = 0.2

# Définir la topologie du modèle.
my_model = create_model(learning_rate)

# Entraînement du modèle
epochs, hist = train_model(my_model, x_train_normalized, y_train,
                           epochs, batch_size, validation_split)

# Évaluation sur l'ensemble de test.
print("\n Évaluer le nouveau modèle sur l'ensemble de test :")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

# Liste des métriques.
list_of_metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
# Tracement des métriques
plot_curve(epochs, hist, list_of_metrics)

# Sauvegarde du modèle pour une réutilisation rapide dans une autre application.
my_model.save('tp1.keras')
