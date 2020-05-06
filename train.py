from dataset import load_data
from decision_tree import decision_tree_train
import numpy as np
from settings import EnsembleConfig


def train(x_train: list, y_train: list, x_validation: list, y_validation: list, config: EnsembleConfig):
    N = len(x_train)
    import random
    if config.classifier_mode == 'DTREE':
        if config.ensemble_mode == 'SINGLE':
            decision_tree_train(x_train, y_train, x_validation, y_validation, config.ensemble_mode)
        elif config.ensemble_mode == 'BAGGING':
            for tree_id in range(0, config.bagging_times):
                x_cur = []
                y_cur = []
                print("Bagging %dth model(%s): " % (tree_id, config.classifier_mode))
                for j in range(0, N):
                    idx = int(random.random() * N)
                    x_cur.append(x_train[idx])
                    y_cur.append(y_train[idx])
                decision_tree_train(np.array(x_cur), np.array(y_cur), x_validation, y_validation, "BAGGING", tree_id)

    elif config.classifier_mode == 'SVM':
        pass
