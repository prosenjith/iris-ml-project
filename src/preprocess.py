import pandas as pd
from sklearn.datasets import load_iris


def load_data():
    """Loads the Iris dataset and returns a pandas DataFrame."""
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data


def preprocess_data(df):
    """Preprocesses the Iris dataset (e.g., normalize, handle missing values)."""
    # Example: Normalize feature values (if needed)
    df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()
    return df
