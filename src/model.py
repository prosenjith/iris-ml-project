from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def split_data(df):
    """Splits the dataset into training and testing sets."""
    X = df.iloc[:, :-1]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Trains a logistic regression model on the dataset."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model