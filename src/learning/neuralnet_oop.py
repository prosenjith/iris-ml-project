from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_shape):
        """
        Initialize the NeuralNetwork class.
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def build_model(self):
        """
        Build a simple neural network with one hidden layer.
        """
        # Define the neural network
        self.model = Sequential([
            Input(shape=self.input_shape),
            Dense(8, activation='relu'),  # Hidden layer
            Dense(1)  # Output layer
        ])
        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error'])
        print("Model Summary:")
        self.model.summary()

    def train(self, X, y, epochs=500, verbose=0):
        """
        Train the neural network model.
        """
        print("Training the model...")
        self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose)
        print("Training complete!")

    def predict(self, X):
        """
        Predict outputs for the given input.
        """
        predictions = self.model.predict(X)
        return predictions

    def plot_loss(self):
        """
        Visualize the training loss over epochs.
        """
        plt.plot(self.history.history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


# Main Program
if __name__ == "__main__":
    # Generate synthetic data
    X_train = np.array([1, 2, 3, 4, 5], dtype=np.float32)  # Input features
    y_train = np.array([1, 3, 5, 7, 9], dtype=np.float32)  # Output labels

    # Initialize the NeuralNetwork
    nn = NeuralNetwork(input_shape=[1])

    # Build, train, and evaluate the model
    nn.build_model()
    nn.train(X_train, y_train, epochs=500)

    # Test the model
    X_test = np.array([6, 7, 8], dtype=np.float32)
    predictions = nn.predict(X_test)
    print("Predictions for [6, 7, 8]:", predictions.flatten())

    # Plot the training loss
    nn.plot_loss()
