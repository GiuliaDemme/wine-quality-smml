import numpy as np
from src.evaluation import accuracy


class LogRegr:
    def __init__(self, learning_rate=1e-2, n_iters=1000, kernel='gaussian', sigma=0.5, tol=1e-6):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.kernel = kernel
        self.sigma = sigma
        self.tol = tol
        self.weights = None  # theta o alpha, a seconda del kernel
        self.X_train = None  # necessario solo per kernel non lineare

    def _get_cls_map(self, y):
        return np.where(y < 6, 0, 1)

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _gaussian_kernel(self, X1, X2):
        # Efficient computation of RBF kernel matrix
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        sq_dists = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-sq_dists / (2 * self.sigma ** 2))

    def _compute_features(self, X):
        if self.kernel == 'linear':
            return self._add_intercept(X)
        if self.kernel == 'gaussian':
            return self._gaussian_kernel(X, self.X_train)

    def fit(self, X, y, save_loss=False):
        y = self._get_cls_map(y)
        if self.kernel != 'linear':
            self.X_train = X

        X = self._compute_features(X)
        self.weights = np.zeros(X.shape[1])
        self.loss_history, self.accuracy_history = [], []
        prev_loss = float('inf')

        for i in range(self.n_iters):
            preds = self.sigmoid(X @ self.weights)
            loss = -np.mean(y * np.log(preds + 1e-8) + (1 - y) * np.log(1 - preds + 1e-8))

            if save_loss:
                acc = accuracy(preds.round(), y)
                self.loss_history.append(loss)
                self.accuracy_history.append(acc)

            if abs(prev_loss - loss) < self.tol:
                print(f'Convergence reached after {i+1} iterations')
                break

            prev_loss = loss

            gradient = X.T @ (preds - y) / len(y)
            self.weights -= self.lr * gradient


    def predict(self, X):
        Z = self._compute_features(X)
        return self.sigmoid(Z @ self.weights).round()