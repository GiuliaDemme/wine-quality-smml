import numpy as np
from src.utils import plot_training_curves
from src.evaluation import accuracy


class SMO:
    def __init__(self, C=5, kernel='gaussian', max_iter=50, tol=1e-6, eps=1e-6, sigma=1):
        self.C = C
        self.b = 0
        self.sigma = sigma
        self.is_linear_kernel = False

        self.kernel = kernel  # 'linear', 'gaussian'
        if kernel == 'linear':
            self.kernel_func = self.linear_kernel
            self.is_linear_kernel = True
        elif kernel == 'gaussian':
            self.kernel_func = self.gaussian_kernel

        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def _get_cls_map(self, y):
        # if y < 6 then map to -1, 1 otherwise
        return np.where(y < 6, -1, 1)

    def hinge_loss(self, X, y):
        """
        Compute hinge loss: (1/n) * sum(max(0, 1 - y*f(x))) + (lambda/2)||w||^2
        In dual (kernel) case, the regularization part is implicit,
        but we approximate using alphas if needed.
        """
        n = X.shape[0]
        # prediction scores f(x)
        scores = np.array([self.prediction(x) for x in X])
        loss = np.mean(np.maximum(0, 1 - y * scores))
        return loss

    def linear_kernel(self, x1, x2, b=0):
        return x1 @ x2.T + b

    def gaussian_kernel(self, x1, x2):
        # case 1D & 1D
        if np.ndim(x1) == 1 and np.ndim(x2) == 1:
            return np.exp(-(np.linalg.norm(x1 - x2, 2)) ** 2 / (2 * self.sigma ** 2))
        # case 1D & batch
        elif (np.ndim(x1) > 1 and np.ndim(x2) == 1) or (np.ndim(x1) == 1 and np.ndim(x2) > 1):
            return np.exp(-(np.linalg.norm(x1 - x2, 2, axis=1) ** 2) / (2 * self.sigma ** 2))
        # case batch & batch
        elif np.ndim(x1) > 1 and np.ndim(x2) > 1:
            return np.exp(-(np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], 2, axis=2) ** 2) / (2 * self.sigma ** 2))
        return 0.

    def prediction(self, x):
        return (self.alphas * self.y) @ self.kernel_func(self.X, x) + self.b

    def predict(self, x):
        return np.sign(self.prediction(x))

    def get_error(self, i):
        return self.prediction(self.X[i, :]) - self.y[i]

    # alphas update
    def take_step(self, i1, i2):
        if i1 == i2:
            return 0

        x1 = self.X[i1, :]
        x2 = self.X[i2, :]

        y1 = self.y[i1]
        y2 = self.y[i2]

        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]

        b = self.b

        E1 = self.get_error(i1)
        E2 = self.get_error(i2)

        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        if L == H:
            return 0

        k11 = self.kernel_func(x1, x1)
        k12 = self.kernel_func(x1, x2)
        k22 = self.kernel_func(x2, x2)

        eta = k11 + k22 - 2 * k12

        if eta > 0:
            # alpha2 update
            alpha2_new = alpha2 + y2 * (E1 - E2) / eta
            # alpha2 clipping
            alpha2_new = np.clip(alpha2_new, L, H)
        else:
            # Abnormal case for eta <= 0, treat this scenario as no progress
            return 0

        # Numerical tolerance
        if abs(alpha2_new - alpha2) < self.eps * (alpha2 + alpha2_new + self.eps):
            return 0

        # alpha1 update
        alpha1_new = alpha1 + y1*y2 * (alpha2 - alpha2_new)

        # Numerical tolerance
        if alpha1_new < self.eps:
            alpha1_new = 0
        elif alpha1_new > (self.C - self.eps):
            alpha1_new = self.C

        # Update threshold
        b1 = b - E1 - y1 * (alpha1_new - alpha1) * k11 - y2 * (alpha2_new - alpha2) * k12
        b2 = b - E2 - y1 * (alpha1_new - alpha1) * k12 - y2 * (alpha2_new - alpha2) * k22
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = 0.5 * (b1 + b2)

        # Update weight vector for linear SVM
        if self.is_linear_kernel:
            self.w = self.w + y1 * (alpha1_new - alpha1) * x1 \
                     + y2 * (alpha2_new - alpha2) * x2

        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        # Error cache update
        # if alpha1 & alpha2 are not at bounds, the error will be 0
        self.error[i1] = 0
        self.error[i2] = 0

        i_list = [idx for idx, alpha in enumerate(self.alphas) if 0 < alpha < self.C]
        for i in i_list:
            self.error[i] += \
                y1 * (alpha1_new - alpha1) * self.kernel_func(x1, self.X[i, :]) \
                + y2 * (alpha2_new - alpha2) * self.kernel_func(x2, self.X[i, :]) \
                + (self.b - b)

        return 1

    # alphas choice
    def examine_example(self, i2):
        y2 = self.y[i2]
        alpha2 = self.alphas[i2]
        E2 = self.get_error(i2)
        r2 = E2 * y2

        # Choose the one that is likely to violate KKT
        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            if len(self.alphas[(0 < self.alphas) & (self.alphas < self.C)]) > 1:
                if E2 > 0:
                    i1 = np.argmin(self.error)
                else:
                    i1 = np.argmax(self.error)

                if self.take_step(i1, i2):
                    return 1

            # loop over all non-zero and non-C alpha, starting at a random point
            i1_list = [idx for idx, alpha in enumerate(self.alphas) if 0 < alpha < self.C]
            i1_list = np.roll(i1_list, np.random.choice(np.arange(self.m)))
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

            # loop over all possible i1, starting at a random point
            i1_list = np.roll(np.arange(self.m), np.random.choice(np.arange(self.m)))
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

        return 0

    def fit(self, X, y):
        self.X = X
        self.y = self._get_cls_map(y)
        self.m, self.n = np.shape(self.X)
        self.alphas = np.zeros(self.m)
        self.error = np.array([self.get_error(i) for i in range(self.m)])
        self.w = np.zeros(self.n)  # used by linear kernel
        self.loss_history, self.accuracy_history = [], []

        loop_num = 0
        numChanged = 0
        examineAll = True
        while numChanged > 0 or examineAll:
            print('ei')
            if loop_num >= self.max_iter:
                print('max')
                break

            numChanged = 0
            if examineAll:
                for i2 in range(self.m):
                    numChanged += self.examine_example(i2)
            else:
                i2_list = [idx for idx, alpha in enumerate(self.alphas) \
                           if 0 < alpha and alpha < self.C]
                for i2 in i2_list:
                    numChanged += self.examine_example(i2)

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True

            loss = self.hinge_loss(self.X, self.y)
            preds = self.predict(self.X)
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy(preds, self.y))

            loop_num += 1

        plot_training_curves(self.loss_history, self.accuracy_history, self.__class__.__name__, self.kernel)