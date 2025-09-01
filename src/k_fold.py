import numpy as np
from src.preprocessing import preprocessing
from src.utils import plot_training_curves
from itertools import product
from src.evaluation import accuracy, precision, recall, f1_score, confusion_matrix
from tqdm import tqdm
from src.preprocessing import standardize, print_class_distribution
import matplotlib.pyplot as plt


# it creates the grid for grid search considering that the sigma parameter is not needed for linear kernel
def parameter_grid(param_dict):
    combos = []
    for kernel in param_dict.get("kernel", []):
        valid_params = {k: v for k, v in param_dict.items()
                        if k != 'kernel' and not (kernel == "linear" and k == "sigma")}

        for values in product(*valid_params.values()):
            combo = dict(zip(valid_params.keys(), values))
            combo["kernel"] = kernel
            combos.append(combo)
    return combos


def kfold_indices(y, k, shuffle=True):
    pos_idx = np.where(y >= 6)[0]
    neg_idx = np.where(y < 6)[0]

    if shuffle:
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)

    folds = []
    for i in range(k):
        test_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.concatenate([np.concatenate(pos_folds[:i] + pos_folds[i + 1:]),
                                    np.concatenate(neg_folds[:i] + neg_folds[i + 1:])])
        if shuffle:
            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)
        folds.append((train_idx, test_idx))

    return folds


def nested_cross_validation(X, y, model_class, param_grid, k_outer, k_inner, metrics):
    outer_folds = kfold_indices(y, k_outer)
    train_errors, test_errors, precisions, recalls, f1s, cms = [], [], [], [], [], []
    loss_history, accuracy_history = [], []
    results = {
        'Train Accuracy': [],
        **{name: [] for name, _ in metrics.items()}
    }

    # OUTER LOOP
    for train_indices, test_indices in outer_folds:
        X_train_outer, y_train_outer = X[train_indices], y[train_indices]
        X_test_outer, y_test_outer = X[test_indices], y[test_indices]
        X_train_outer, X_test_outer = standardize(X_train_outer, X_test_outer)

        inner_folds = kfold_indices(y_train_outer, k_inner)
        best_score = float('inf')
        best_params = None

        # INNER LOOP
        for i, param_comb in enumerate(tqdm(parameter_grid(param_grid))):
            scores = []

            for inner_train_indices, val_indices in inner_folds:
                X_train_inner, y_train_inner = X_train_outer[inner_train_indices], y_train_outer[inner_train_indices]
                X_val_inner, y_val_inner = X_train_outer[val_indices], y_train_outer[val_indices]

                model = model_class(**param_comb)
                model.fit(X_train_inner, y_train_inner)
                y_val_pred = model.predict(X_val_inner)
                score = 1 - accuracy(model._get_cls_map(y_val_inner), y_val_pred)  # error
                scores.append(score)

            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = param_comb

        # OUTER LOOP: retrain on full outer training with best params
        print("Best parameters are:", best_params)
        best_model = model_class(**best_params)
        best_model.fit(X_train_outer, y_train_outer, save_loss=True)

        loss_history.append(best_model.loss_history)
        accuracy_history.append(best_model.accuracy_history)

        y_train_pred = best_model.predict(X_train_outer)
        y_test_pred = best_model.predict(X_test_outer)
        y_test_outer = best_model._get_cls_map(y_test_outer)

        train_accuracy = accuracy(best_model._get_cls_map(y_train_outer), y_train_pred)

        results['Train Accuracy'].append(train_accuracy)
        train_errors.append(1 - train_accuracy)
        for name, func in metrics.items():
            if name == 'Test Accuracy':
                acc = func(y_test_outer, y_test_pred)
                test_error = 1 - acc
                results[name].append(acc)
                test_errors.append(test_error)
            else:
                results[name].append(func(y_test_outer, y_test_pred))

    plot_training_curves(np.mean(np.array(loss_history), axis=0), np.mean(np.array(accuracy_history), axis=0),
                         model_class.__name__)

    print('Test errors across outer folds: ', test_errors)

    results = {
        name: (np.sum(metrics, axis=0) if name == "Confusion_matrix" else np.mean(metrics))
        for name, metrics in results.items()
    }

    return results


if __name__ == '__main__':
    k = 5
    X, y = preprocessing()
    fold_indices = kfold_indices(X, k)




