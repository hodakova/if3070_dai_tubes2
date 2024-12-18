from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

from KNearestNeighbor import KNearestNeighbor
import numpy as np


def test_knn():
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNearestNeighbor()
    knn.fit(X_train, y_train, max_k=10, num_of_k_values=5)

    predictions = knn.predict(X_test)
    print("Test Predictions:", predictions)
    print("Actual Labels:   ", y_test.tolist())
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print('============================================================================\n')

    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)
    knn = KNearestNeighbor()
    knn.fit(X_train, y_train, max_k=10, num_of_k_values=5)
    predictions = knn.predict(X_test)
    print("Test Predictions:", predictions)
    print("Actual Labels:   ", y_test.tolist())
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    test_knn()
