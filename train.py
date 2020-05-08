from dataset import load_data
from decision_tree import decision_tree_train
from svm import svm_train
import numpy as np
import os
from settings import EnsembleConfig, check_dir_exist


def train(x_train: list, y_train: list, x_validation: list, y_validation: list, config: EnsembleConfig):
    N = len(x_train)
    import random
    train_func = decision_tree_train
    if config.classifier_mode == 'DTREE':
        train_func = decision_tree_train
    elif config.classifier_mode == 'SVM':
        train_func = svm_train
    check_dir_exist(config)
    if config.ensemble_mode == 'SINGLE':
        train_func(x_train, y_train, x_validation, y_validation, config)
    elif config.ensemble_mode == 'BAGGING':
        for id in range(0, config.bagging_times):
            x_cur = []
            y_cur = []
            print("Bagging %dth model(%s): " % (id, config.classifier_mode))
            for j in range(0, N):
                idx = int(random.random() * N)
                x_cur.append(x_train[idx])
                y_cur.append(y_train[idx])
            train_func(np.array(x_cur), np.array(y_cur), x_validation, y_validation, config, id)
    elif config.ensemble_mode == 'ADA_BOOST_M1':
        sample_weights = np.array([1.0 / len(x_train)] * len(x_train))
        i = 0
        while i < config.ada_times:
            print("AdaBoostM1 %dth model(%s): " % (i, config.classifier_mode))
            sample_idxs = np.random.choice(x_train.shape[0], size=x_train.shape[0], p=sample_weights)
            x_sample = x_train[sample_idxs]
            y_sample = y_train[sample_idxs]
            sample_weights, err = train_func(x_sample, y_sample, x_validation, y_validation, config, i,
                                             sample_weights, x_train, y_train)
            if err > 0.6:
                break
            i += 1