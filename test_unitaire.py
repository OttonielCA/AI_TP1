import unittest
import tensorflow as tf


class TestMNISTModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Charger les données MNIST
        (cls.x_train, cls.y_train), (cls.x_test, cls.y_test) = tf.keras.datasets.mnist.load_data()
        cls.x_train, cls.x_test = cls.x_train / 255.0, cls.x_test / 255.0

        # Charger ou construire le modèle
        cls.model = tf.keras.models.load_model('tp1.keras')


    def test_model_structure(self):
        # Vérifier que le modèle a le bon nombre de couches
        self.assertEqual(len(self.model.layers), 5)

    def test_training(self):
        # Entraîner le modèle sur un petit échantillon et vérifier si l'entraînement fonctionne
        history = self.model.fit(self.x_train[:1000], self.y_train[:1000], epochs=1)
        self.assertGreater(history.history['accuracy'][-1], 0.5)

    def test_evaluation(self):
        # Évaluer le modèle sur un petit échantillon et vérifier la précision
        loss, accuracy = self.model.evaluate(self.x_test[:1000], self.y_test[:1000])
        self.assertGreater(accuracy, 0.5)


if __name__ == '__main__':
    unittest.main()
