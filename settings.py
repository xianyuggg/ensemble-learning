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
        elif self.ensemble_mode == 'ADA':
            return 'ADA-' + self.ensemble_mode + '-' + str(self.ada_times)
