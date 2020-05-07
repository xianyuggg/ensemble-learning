class EnsembleConfig:
    def __init__(self, bagging_times, ada_times, classifier_mode, ensemble_mode):
        """
        :param bagging_times:
        :param ada_times:
        :param classifier_mode: DTREE, SVM
        :param ensemble_mode: BAGGING, ADA_BOOST_M1, SINGLE
        """
        self.bagging_times = bagging_times
        self.ada_times = ada_times
        self.classifier_mode = classifier_mode
        self.ensemble_mode = ensemble_mode

    def __str__(self):
        if self.ensemble_mode == 'BAGGING':
            return 'BAGGING-' + self.classifier_mode + '-' + str(self.bagging_times)
        elif self.ensemble_mode == 'ADA_BOOST_M1':
            return 'ADA-' + self.classifier_mode + '-' + str(self.ada_times)


import numpy as np


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def get_max_appear_num(array: list):
    counts = np.bincount(array)
    # 返回众数
    return np.argmax(counts)
