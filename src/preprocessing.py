import pandas as pd
import numpy as np
import os
from src.utils import plot_target_distribution, plot_correlation_matrix

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)


def dataset_loading():
    dataframes = [pd.read_csv(f"../data/{file_name}", sep=";") for file_name in os.listdir('../data')]
    dataset = pd.concat(dataframes, ignore_index=True)
    return dataset


def features_target_sep(dataset, target_name):
    features = dataset.drop(columns=target_name, axis=1)
    target = dataset[target_name]
    return features.values, target.values


def shuffle_data(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(y, train_size=0.5, shuffle=True):
    pos_idx = np.where(y >= 6)[0]
    neg_idx = np.where(y < 6)[0]

    if shuffle:
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

    split_pos = int(train_size * len(pos_idx))
    split_neg = int(train_size * len(neg_idx))

    train_idx = np.concatenate([pos_idx[:split_pos], neg_idx[:split_neg]])
    test_idx = np.concatenate([pos_idx[split_pos:], neg_idx[split_neg:]])

    if shuffle:
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)

    return train_idx, test_idx


def print_class_distribution(y, label="Dataset"):
    y_bin = np.array(y) >= 6
    unique, counts = np.unique(y_bin, return_counts=True)
    distribution = dict(zip(unique, counts))
    total = len(y)
    print(f"{label} distribution:")
    for cls, count in distribution.items():
        print(f"  Class {cls}: {count} samples ({count/total:.2%})")
    print()


def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Avoid zero-divisions
    std[std == 0] = 1

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_test_std


def preprocessing():
    dataset = dataset_loading()
    target_column = dataset.columns[-1]
    X, y = features_target_sep(dataset, target_column)
    train_idx, test_idx = train_test_split(y, train_size=0.8)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    # standardize
    X_train_std, X_test_std = standardize(X_train, X_test)
    return X_train_std, X_test_std, y_train, y_test, X, y


if __name__ == '__main__':
    winequality_dataset = dataset_loading()

    # name of the target column
    target_column = winequality_dataset.columns[-1]

    # print first 5 rows
    print(winequality_dataset.head())

    print(200 * '-')

    # number of rows and columns in the dataset
    print(f"The shape of the dataset is {winequality_dataset.shape}")

    print(200 * '-')

    # getting the statistical measures of the dataset
    print(winequality_dataset.describe())

    print(200 * '-')

    # values distribution of the target column
    print(winequality_dataset[target_column].value_counts())

    print(200 * '-')

    # visualization of the distribution of the target column
    plot_target_distribution(winequality_dataset[target_column])

    # visualization of the heatmap
    plot_correlation_matrix(winequality_dataset)

    # find if there are missing values
    print(f"Missing values for each feature:\n{winequality_dataset.isna().sum()}")

    print(200 * '-')

    # features-target separation
    X, y = features_target_sep(winequality_dataset, target_column)

    # Class distribution before the split
    print_class_distribution(y, "Original dataset")

    # distributed train-test split
    train_idx, test_idx = train_test_split(y, train_size=0.8, shuffle=True)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    print(y.shape, X_train.shape, X_test.shape)

    print(200 * '-')

    # Class distribution after the split
    print_class_distribution(y_train, "Training set")
    print_class_distribution(y_test, "Test set")

    print(200 * '-')

    # Standardization
    X_train_std, X_test_std = standardize(X_train, X_test)
    print(pd.DataFrame(X_train_std).describe())
    print(pd.DataFrame(X_test_std).describe())



