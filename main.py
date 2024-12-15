from src.preprocess import load_data, preprocess_data
from src.model import split_data, train_model
from src.evaluate import evaluate_model

def main():
    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Display results
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()