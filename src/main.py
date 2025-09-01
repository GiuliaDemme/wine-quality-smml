from src.preprocessing import preprocessing
from src.k_fold import nested_cross_validation
from src.evaluation import accuracy, precision, recall, f1_score, confusion_matrix
from src.config import *
from src.utils import plot_confusion_matrix
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)


if __name__ == '__main__':

    # preprocessing
    X_train, X_test, Y_train, Y_test, features, target = preprocessing()

    metrics = {
        'Test Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1_score,
        'Confusion_matrix': confusion_matrix
    }

    for model_params in model_grid:
        print(model_params['model'].__name__)

        if model_params['kfold']:
            print(f'Evaluating {model_params["model"].__name__} with k-fold cross-validation')
            results = nested_cross_validation(features, target, model_params['model'], model_params['grid'],
                                              k_outer, k_inner, metrics)

        else:
            print(f'Evaluating {model_params["model"].__name__} with {model_params["grid"]}')
            model = model_params['model'](**model_params['grid'])
            # training the model
            model.fit(X_train, Y_train)
            # making prediction
            X_train_prediction = model.predict(X_train)
            X_test_prediction = model.predict(X_test)
            Y_test = model._get_cls_map(Y_test)

            results = {
                'Train Accuracy': accuracy(X_train_prediction, model._get_cls_map(Y_train)),
                **{name: func(X_test_prediction, Y_test) for name, func in metrics.items()}
            }

        print(results)
        plot_confusion_matrix(results['Confusion_matrix'], model_params)

