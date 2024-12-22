import numpy as np
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class KNearestNeighbor:
    def __init__(self, verbose=True):
        self.best_k = None
        self.training_data = None
        self.training_labels = None
        self.verbose = verbose

    def fit(self, X_train, Y_train, max_k=None, num_of_k_values=10):
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        self.training_data = X_train
        self.training_labels = Y_train

        n_splits = 5
        split_length = len(X_train) // n_splits

        if max_k is None:
            max_k = int(np.sqrt(len(self.training_data)))

        k_range = np.linspace(1, max_k, num_of_k_values, dtype=int)
        k_accuracies = {k: [] for k in k_range}

        if self.verbose:
            logging.info(f"Starting fit with max_k={max_k} and num_of_k_values={num_of_k_values}")
            logging.info(f"Training data shape: {X_train.shape}, Labels shape: {Y_train.shape}")

        for fold in range(n_splits):
            val_start = fold * split_length
            val_end = (fold + 1) * split_length if fold != n_splits - 1 else len(X_train)

            # Ensure slices are numpy arrays
            X_val = X_train[val_start:val_end]
            Y_val = Y_train[val_start:val_end]
            X_fold_train = np.vstack((X_train[:val_start], X_train[val_end:]))
            Y_fold_train = np.hstack((Y_train[:val_start], Y_train[val_end:]))

            for k in k_range:
                predictions = self._predict_vectorized(X_val, X_fold_train, Y_fold_train, k=k)
                accuracy = np.mean(predictions == Y_val)
                k_accuracies[k].append(accuracy)

                if self.verbose:
                    logging.info(f"Fold {fold + 1}/{n_splits}, k={k}, accuracy={accuracy:.4f}")

        avg_k_accuracies = {k: np.mean(acc) for k, acc in k_accuracies.items()}
        best_k = max(avg_k_accuracies, key=avg_k_accuracies.get)
        self.best_k = (best_k, avg_k_accuracies[best_k])

        if self.verbose:
            logging.info(f"Best k: {self.best_k[0]} with average training CV accuracy: {self.best_k[1]:.4f}")

        return self

    def _compute_distances(self, X_test, train_data):
        X_test = np.array(X_test)
        train_data = np.array(train_data)

        try:
            dists = np.sqrt(
                np.maximum(
                    np.sum(X_test[:, np.newaxis, :] ** 2, axis=2) +
                    np.sum(train_data[np.newaxis, :, :] ** 2, axis=2) -
                    2 * np.dot(X_test, train_data.T),
                    0
                )
            )
        except ValueError as e:
            logging.error(f"Error in distance computation: {e}")
            logging.error(f"X_test shape: {X_test.shape}, train_data shape: {train_data.shape}")
            raise

        if self.verbose:
            logging.info(f"Computed distances with shape: {dists.shape}")
        return dists

    def _get_neighbours(self, dists, k):
        nearest_indices = np.argpartition(dists, k, axis=1)[:, :k]

        if self.verbose:
            logging.info(f"Identified {k} nearest neighbors for each test point")
        return nearest_indices

    def _predict_vectorized(self, X_test, train_data=None, train_labels=None, k=None):
        train_data = train_data if train_data is not None else self.training_data
        train_labels = train_labels if train_labels is not None else self.training_labels

        if self.best_k is not None and k is None:
            k = self.best_k[0]

        if self.verbose:
            logging.info(f"Predicting with k={k}, test data shape: {X_test.shape}")

        dists = self._compute_distances(X_test, train_data)
        nearest_indices = self._get_neighbours(dists, k)
        nearest_labels = train_labels[nearest_indices]

        predictions = [Counter(neighbors).most_common(1)[0][0] for neighbors in nearest_labels]

        if self.verbose:
            logging.info(f"Prediction complete for {len(X_test)} test points")

        return np.array(predictions)

    def predict(self, X_test, train_data=None, train_labels=None, k=None):
        return self._predict_vectorized(X_test, train_data, train_labels, k)
