import numpy as np
from collections import Counter

class KNearestNeighbor:
    def __init__(self):
        self.best_k = None
        self.training_data = None
        self.training_labels = None

    def fit(self, X_train, Y_train, max_k=None, num_of_k_values=10):
        self.training_data = np.array(X_train)
        self.training_labels = np.array(Y_train)

        #5FoldCV
        n_splits = 5
        split_length = len(X_train) // n_splits
        if max_k is None:
            max_k = int(np.sqrt(len(self.training_data)))
        k_range = np.linspace(1, max_k, num_of_k_values, dtype=int)
        k_accuracies = {k: [] for k in k_range}

        for fold in range(n_splits):
            val_start = fold * split_length
            val_end = (fold + 1) * split_length if fold != n_splits - 1 else len(X_train)
            X_val = X_train[val_start:val_end]
            Y_val = Y_train[val_start:val_end]
            X_fold_train = np.vstack((X_train[:val_start], X_train[val_end:]))
            Y_fold_train = np.hstack((Y_train[:val_start], Y_train[val_end:]))

            for k in k_range:
                predictions = self.predict(X_val, X_fold_train, Y_fold_train, k)
                accuracy = np.mean(predictions == Y_val)
                k_accuracies[k].append(accuracy)

        avg_k_accuracies = {k: np.mean(acc) for k, acc in k_accuracies.items()}
        best_k = max(avg_k_accuracies, key=avg_k_accuracies.get)
        self.best_k = (best_k, avg_k_accuracies[best_k])

        print(f'k value={self.best_k[0]} is used with the best avg_training_cv={self.best_k[1]}')
        return self

    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _get_neighbours(self, test_point, train_data, k):
        distances = [self._euclidean_distance(test_point, train_point) for train_point in train_data]
        neighbors_indices = np.argsort(distances)[:k]
        return neighbors_indices

    def predict(self, X_test, train_data=None, train_labels=None, k=None):
        train_data = train_data if train_data is not None else self.training_data
        train_labels = train_labels if train_labels is not None else self.training_labels

        if self.best_k is not None and k is None:
            k = self.best_k[0]

        predictions = []
        for test_point in X_test:
            neighbors_indices = self._get_neighbours(test_point, train_data, k)
            neighbor_labels = train_labels[neighbors_indices]
            most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return predictions
