from sklearn import tree
import joblib
import numpy as np
import os
from settings import EnsembleConfig, get_model_dir
from sklearn.metrics import mean_squared_error, accuracy_score
import math

def decision_tree_train(x_train, y_train, x_validation, y_validation, config: EnsembleConfig, tree_id: int = '',
                        sample_weights=None, raw_x_train=None, raw_y_train=None):
    model = tree.DecisionTreeClassifier(class_weight='balanced')

    model.fit(x_train, y_train)
    if not os.path.exists('model'):
        os.mkdir('model')

    if config.ensemble_mode == 'BAGGING':
        joblib.dump(model, get_model_dir(config) + 'dtree_' + str(tree_id) + '.pkl')
    elif config.ensemble_mode == 'ADA_BOOST_M1':
        raw_pred = model.predict(raw_x_train)
        err = 1. * np.dot(np.array(raw_pred) != np.array(raw_y_train), sample_weights)
        print("current model(%s) err: " % str(config), err)
        print("current model(%s) acc on training set: " % str(config), 1 - accuracy_score(raw_pred, raw_y_train))
        print("current model(%s) acc on validation: " % str(config), 1 - accuracy_score(model.predict(x_validation), y_validation))
        if err > 0.6:
            return sample_weights, err
        beta = err / (1.3 - err)
        update_weights = [1 if raw_y_train[i] != raw_pred[i] else beta for i in range(0, len(raw_x_train))]
        sample_weights = np.multiply(sample_weights, update_weights)
        sample_weights = sample_weights / np.sum(sample_weights)  # normalization
        joblib.dump(model, get_model_dir(config) + 'dtree_' + str(tree_id) + '.pkl')
        with open(get_model_dir(config) + 'beta_' + str(tree_id) + '.txt', 'w') as f:
            f.write(str(beta))
        print("current model(%s) rmse: " % str(config),
              math.sqrt(mean_squared_error(model.predict(x_validation), y_validation)))
        return sample_weights, err
    elif config.ensemble_mode == "SINGLE":
        joblib.dump(model, get_model_dir(config) + 'dtree.pkl')
    else:
        print("unimplemented in decision_tree_train!")
        exit(0)

    print("current model(%s) rmse: " % str(config),
          math.sqrt(mean_squared_error(model.predict(x_validation), y_validation)))


def decision_tree_predict(words_data, config: EnsembleConfig, model_id=''):
    if config.ensemble_mode == 'BAGGING':
        model = joblib.load(get_model_dir(config) + 'dtree_' + str(model_id) + '.pkl')
        result = model.predict(words_data)
    elif config.ensemble_mode == "ADA_BOOST_M1":
        model = joblib.load(get_model_dir(config) + 'dtree_' + str(model_id) + '.pkl')
        result = model.predict(words_data)
    elif config.ensemble_mode == 'SINGLE':
        model = joblib.load(get_model_dir(config) + 'dtree.pkl')
        result = model.predict(words_data)
    else:
        print("unimplemented in decision_tree_predict!")
        exit(0)
    return result
