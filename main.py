from dataset import divide_set, load_data, load_test_data
from train import train
from predict import validation
from settings import EnsembleConfig


# DTREE, SVM
# BAGGING, ADA_BOOST_M1, SINGLE
x_train, y_train, x_validation, y_validation, x_test = load_data()
config = EnsembleConfig(bagging_times=10, ada_times=20, classifier_mode='DTREE', ensemble_mode='BAGGING')
# train(x_train, y_train, x_validation, y_validation, config)

validation(x_validation, y_validation, config)
