import unittest
import tensorflow as tf

# Classe pour effectuer le test unitaire
class TestMNISTModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):  # Fonction pour effectuer le chargement du model MNIST
        # Charger les données MNIST a partir de tensorflow
        (cls.x_train, cls.y_train), (cls.x_test, cls.y_test) = tf.keras.datasets.mnist.load_data()
        cls.x_train, cls.x_test = cls.x_train / 255.0, cls.x_test / 255.0

        # Chargement du modèle tp1.keras
        cls.model = tf.keras.models.load_model('tp1.keras')

    # Fonction pour tester si le bon nombre de couches du réseau de neuronnes à été chargé
    def test_model_structure(self):
        # Vérifier que le modèle a le bon nombre de couches
        self.assertEqual(len(self.model.layers), 5)
    # Fonction pour tester si l'entrainement du modèle à bien été compilé
    def test_training(self):
        # Entraîner le modèle sur un petit échantillon et vérifier si l'entraînement fonctionne
        history = self.model.fit(self.x_train[:1000], self.y_train[:1000], epochs=1)
        self.assertGreater(history.history['accuracy'][-1], 0.5)
    # Fonction pour tester si l'évaluation du modèle fonctionne correctement
    def test_evaluation(self):
        # Évaluer le modèle sur un petit échantillon et vérifier la précision
        loss, accuracy = self.model.evaluate(self.x_test[:1000], self.y_test[:1000])
        self.assertGreater(accuracy, 0.5)

# Execution du test unitaire
if __name__ == '__main__':
    unittest.main()
