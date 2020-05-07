from dataset import divide_set, load_data, load_test_data
from train import train
from predict import validation, test
from settings import EnsembleConfig


# DTREE, SVM
# BAGGING, ADA_BOOST_M1, SINGLE
config = EnsembleConfig(bagging_times=10, ada_times=20, classifier_mode='DTREE', ensemble_mode='ADA_BOOST_M1', external_w2v=False, tf_idf=False)

# raw_data can be found in raw_data/test.csv, train.csv...
divide_set(config)

# dataset path can be found in dataset/x_train.npy...
x_train, y_train, x_validation, y_validation, x_test = load_data()

train(x_train, y_train, x_validation, y_validation, config)

validation(x_validation, y_validation, config)

test(x_test, config)
