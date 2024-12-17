from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

def _build_neural_network():
    X = np.array([1, 2, 3, 4, 5], dtype=np.float32)  # Input features
    y = np.array([1, 3, 5, 7, 9], dtype=np.float32)  # Output labels
    print("Input:", X)
    print("Labels:", y)

    # Define the neural network
    model = Sequential([
        Input(shape=(1,)),
        Dense(8, activation='relu'),  # Hidden layer
        Dense(1)  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    # Print model summary
    model.summary()

    # Train the model
    history = model.fit(X, y, epochs=500, verbose=0)

    # Display training progress
    print("Model trained successfully!")

    # Predict values
    test_input = np.array([6, 7, 8], dtype=np.float32)
    predictions = model.predict(test_input)

    print("Predictions for input [6, 7, 8]:", predictions)

    # Plot training loss
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    _build_neural_network()
