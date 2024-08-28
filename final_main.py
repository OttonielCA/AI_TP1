import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def show_mnist_graphic_number(img):
    img = np.array(img, dtype='float')
    pixels = img.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Load model
new_model = tf.keras.models.load_model('tp1.keras')
# Show the model architecture
new_model.summary()

# Show the prediction asked by the teacher
img = [1, 6, 3513, 145, 458]
for x in img:
    img1 = x_test[x]
    print(new_model.predict(np.reshape(img1, (1, 28, 28))))
    show_mnist_graphic_number(img1)
