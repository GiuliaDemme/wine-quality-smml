from src.models.svm import SMO
from src.models.log_regr import LogRegr

# k-fold cross-validation
k_outer = 5
k_inner = 5

# hyperparameters grids
svm_param_grid = {
    "C": [1e-3, 1e-2, 1e-1, 1],
    "kernel": ['linear', 'gaussian'],
    "sigma": [0.1, 0.5, 1., 5., 10.]            # only for gaussian kernel
}

logregr_param_grid = {
    "learning_rate": [1e-2, 1e-1, 1, 10],
    "kernel": ['linear', 'gaussian'],
    "sigma": [0.1, 0.5, 1., 5.]            # only for gaussian kernel
}


model_grid = [
    {'model': SMO, 'grid': {'kernel': 'linear', 'C': 0.5}, 'kfold': False},
    {'model': SMO, 'grid': {'kernel': 'gaussian', 'C': 3, 'sigma': 0.5}, 'kfold': False},
    {'model': LogRegr, 'grid': logregr_param_grid, 'kfold': True}]
