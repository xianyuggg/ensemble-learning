from sklearn.svm import SVC, LinearSVC
import os
import joblib
import numpy as np
from settings import get_model_dir, EnsembleConfig
from sklearn.metrics import mean_squared_error
import math


def svm_train(x_train, y_train, x_validation, y_validation, config: EnsembleConfig, svm_id: int = '',
              sample_weights=None, raw_x_train=None, raw_y_train=None):
    # Prefer dual=False when n_samples > n_features.
    model = LinearSVC(multi_class='ovr', class_weight='balanced', verbose=True, dual=False, max_iter=1000)
    model.fit(x_train, y_train)
    if config.ensemble_mode == 'BAGGING':
        joblib.dump(model, get_model_dir(config) + 'svm_' + str(svm_id) + '.pkl')
    elif config.ensemble_mode == 'ADA_BOOST_M1':
        raw_pred = model.predict(raw_x_train)
        err = 1. * np.dot(np.array(raw_pred) != np.array(raw_y_train), sample_weights)
        if err > 0.5:
            return sample_weights, err
        beta = err / (1.0 - err)
        update_weights = [1 if raw_y_train[i] != raw_pred[i] else beta for i in range(0, len(raw_x_train))]
        sample_weights = np.multiply(sample_weights, update_weights)
        sample_weights = sample_weights / np.sum(sample_weights)  # normalization
        joblib.dump(model, get_model_dir(config) + 'svm_' + str(svm_id) + '.pkl')
        with open(get_model_dir(config) + 'beta_' + str(svm_id) + '.txt', 'w') as f:
            f.write(str(beta))
        print("current model(%s) rmse: " % str(config),
              math.sqrt(mean_squared_error(model.predict(x_validation), y_validation)))

        return sample_weights, err
    elif config.ensemble_mode == "SINGLE":
        joblib.dump(model, get_model_dir(config) + 'svm_model.pkl')
    else:
        print("unimplemented in svm_train!")
        exit(0)
    print("current model(%s) rmse: "% str(config), math.sqrt(mean_squared_error(model.predict(x_validation), y_validation)))

def svm_predict(words_data, config: EnsembleConfig, model_id: int = None):
    if config.ensemble_mode == 'BAGGING':
        model = joblib.load(get_model_dir(config) + 'svm_' + str(model_id) + '.pkl')
        result = model.predict(words_data)
    elif config.ensemble_mode == "ADA_BOOST_M1":
        model = joblib.load(get_model_dir(config) + 'svm_' + str(model_id) + '.pkl')
        result = model.predict(words_data)
    elif config.ensemble_mode == 'SINGLE':
        model = joblib.load(get_model_dir(config) + 'svm_model.pkl')
        result = model.predict(words_data)
    else:
        print("unimplemented in svm_predict!")
        exit(0)
    return result
