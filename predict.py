from settings import EnsembleConfig
import numpy as np


def get_max_appear_num(array: list):
    counts = np.bincount(array)
    # 返回众数
    return np.argmax(counts)


def predict(x_validation, config: EnsembleConfig, model_id):
    from decision_tree import decision_tree_predict
    if config.classifier_mode == 'DTREE':
        return decision_tree_predict(x_validation, config.ensemble_mode, model_id)
    elif config.classifier_mode == 'SVM':
        return []
    return []


def validation(x_validation, y_validation, config: EnsembleConfig):
    from sklearn.metrics import accuracy_score
    if config.ensemble_mode == 'BAGGING':
        res = []
        for model_id in range(0, config.bagging_times):
            res.append(predict(x_validation, config, model_id))
        # get most frequent rating
        res = np.array(res).T
        y_predict = [get_max_appear_num(item) for item in res]
        print(str(config) + ' acc on validation set: ', accuracy_score(y_validation, y_predict))

    elif config.ensemble_mode == 'ADA_BOOST_M1':
        pass
