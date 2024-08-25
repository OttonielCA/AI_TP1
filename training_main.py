import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Loading MNIST dataset for train & test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data cleaning/Normalize the training set to be a value between 0 & 1
x_train_normalized = x_train / 255
x_test_normalized = x_test / 255
# print(x_train_normalized[2900]) # Output example 2900 out of training set

# Creating the deep neural network
def create_model(my_learning_rate):
    # Model Init
    model = tf.keras.models.Sequential()
    # Flatten layer
    model.add(layers.Flatten(input_shape=(28, 28)))
    # First hidden layer with 256 nodes
    model.add(layers.Dense(units=256, activation='relu'))
    # Dropout layer with rate 0.4
    model.add(layers.Dense(units=128, activation='relu'))
    # Dropout layer with rate 0.2
    model.add(layers.Dropout(rate=0.2))
    # Output layer
    model.add(layers.Dense(units=10, activation='softmax'))
    # Model compilation
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model

# Training the deep neural network
def train_model(model, train_features, train_label, epochs, batch_size=None, validation_split=0.1):
    # Training model
    history = model.fit(x=train_features, y=train_label, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=validation_split)
    # To track the progression of training
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


def plot_curve(epochs, hist, list_of_metrics):
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and Validation Metrics")  # Add a title for the plot

    for m in list_of_metrics:
        # Show accuracy as 0 to 100% for better readability
        if 'accuracy' in m or 'val_accuracy' in m:
            x = hist[m] * 100
        else:
            x = hist[m]

        # Plotting the metric
        plt.plot(epochs, x, label=m)

    plt.legend(loc='best')  # Place the legend in the best location
    plt.grid(True)  # Add grid lines for better readability
    plt.savefig('metrics_plot.png')  # Save the plot with a fixed filename
    plt.show()


learning_rate = 0.01
epochs = 25
batch_size = 4000
validation_split = 0.2

# Establish the model's topography.
my_model = create_model(learning_rate)

# Training the model
epochs, hist = train_model(my_model, x_train_normalized, y_train,
                           epochs, batch_size, validation_split)

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

# List the metrics you want to plot.
list_of_metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
# Plot the metrics
plot_curve(epochs, hist, list_of_metrics)

# Save model for quick reuse in another app (without training again)
my_model.save('tp1.keras')
