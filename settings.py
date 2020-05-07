class EnsembleConfig:
    def __init__(self, bagging_times, ada_times, classifier_mode, ensemble_mode, external_w2v: bool = False, tf_idf: bool = False):
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
        self.external_w2v = external_w2v
        self.tf_idf = tf_idf

    def __str__(self):
        if self.ensemble_mode == 'BAGGING':
            return 'BAGGING-' + self.classifier_mode + '-' + str(self.bagging_times) + '-ExternalW2V_' + str(
                self.external_w2v) + '-TFIDF_' + str(self.tf_idf)
        elif self.ensemble_mode == 'ADA_BOOST_M1':
            return 'ADA-' + self.classifier_mode + '-' + str(self.ada_times) + '-ExternalW2V_' + str(self.external_w2v) + '-TFIDF_' + str(self.tf_idf)
        elif self.ensemble_mode == 'SINGLE':
            return 'SINGLE-' + self.classifier_mode + '-ExternalW2V_' + str(self.external_w2v) + '-TFIDF_' + str(self.tf_idf)
        exit()


def check_dir_exist(config: EnsembleConfig):
    import os
    if not os.path.exists('model'):
        os.mkdir('model')
    if os.path.exists('model/%s/' % config.ensemble_mode + str(config)):
        files = os.listdir('model/%s/' % config.ensemble_mode + str(config))
        for filename in files:
            os.remove('model/%s/' % config.ensemble_mode + str(config) + '/' + filename)
    if not os.path.exists('model/%s/' % config.ensemble_mode):
        os.mkdir('model/%s/' % config.ensemble_mode)
    if not os.path.exists('model/%s/' % config.ensemble_mode + str(config)):
        os.mkdir('model/%s/' % config.ensemble_mode + str(config))
    return


def get_model_dir(config: EnsembleConfig):
    return 'model/%s/' % config.ensemble_mode + str(config) + '/'


import numpy as np


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def get_max_appear_num(array: list):
    counts = np.bincount(array)
    # 返回众数
    return np.argmax(counts)
