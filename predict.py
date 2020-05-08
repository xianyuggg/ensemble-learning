from settings import EnsembleConfig, get_max_appear_num
import numpy as np
import os
import math
from settings import get_model_dir


def predict(x_validation, config: EnsembleConfig, model_id=None):
    from decision_tree import decision_tree_predict
    from svm import svm_predict
    if config.classifier_mode == 'DTREE':
        return decision_tree_predict(x_validation, config, model_id)
    elif config.classifier_mode == 'SVM':
        return svm_predict(x_validation, config, model_id)
    return []


def validation(x_validation, y_validation, config: EnsembleConfig):
    from sklearn.metrics import accuracy_score
    if config.ensemble_mode == 'BAGGING':
        res = []
        for model_id in range(0, config.bagging_times):
            res.append(predict(x_validation, config, model_id))
        # get most frequent rating
        res = np.array(res).T
        y_predict = [np.mean(item) for item in res]
        # print(str(config) + ' acc on validation set: ', accuracy_score(y_validation, y_predict))

    elif config.ensemble_mode == 'ADA_BOOST_M1':
        files = os.listdir(get_model_dir(config))
        weight = np.array([0.0 for _ in range(0, config.ada_times)])
        ada_len = 0
        for filename in files:
            if filename.split('.')[-1] == 'txt':
                idx = int(filename.split('.')[0].split('_')[1])
                if idx + 1 > ada_len:
                    ada_len = idx + 1
                with open(get_model_dir(config) + filename) as file:
                    weight[idx] = 0.5 * math.log(1 / float(file.read()))

        weight = weight[0:ada_len]
        res = []
        for model_id in range(0, ada_len):
            res.append(predict(x_validation, config, model_id))
        res = np.array(res).T
        y_predict = []
        for i in range(0, len(res)):
            avg = 0.0
            for j in range(0, ada_len):
                avg += (weight[j] * res[i][j])/np.sum(weight)
            y_predict.append(avg)
        # used for calculate accuracy
        # for i in range(0, len(res)):
        #     tmp = [0, 0, 0, 0, 0, 0] # [1][2][3][4][5] is useful
        #     for j in range(0, ada_len):
        #         tmp[res[i][j]] += weight[j]
        #     y_predict.append(np.argmax(tmp))
        # print(str(config), ' acc on validation set: ', accuracy_score(y_validation, y_predict))
    elif config.ensemble_mode == 'SINGLE':
        y_predict = predict(x_validation, config)
    from sklearn.metrics import mean_squared_error
    print(str(config), ' rmse on validation set: ', math.sqrt(mean_squared_error(y_predict, y_validation)))


def test(x_test, config: EnsembleConfig):
    y_predict = []
    if config.ensemble_mode == 'BAGGING':
        res = []
        for model_id in range(0, config.bagging_times):
            res.append(predict(x_test, config, model_id))
        # get most frequent rating
        res = np.array(res).T
        y_predict = [np.mean(item) for item in res]
        # y_predict = [get_max_appear_num(item) for item in res]

    elif config.ensemble_mode == 'ADA_BOOST_M1':
        files = os.listdir(get_model_dir(config))
        weight = np.array([0.0 for _ in range(0, config.ada_times)])
        ada_len = 0
        for filename in files:
            if filename.split('.')[-1] == 'txt':
                idx = int(filename.split('.')[0].split('_')[1])
                if idx + 1 > ada_len:
                    ada_len = idx + 1
                with open(get_model_dir(config) + filename) as file:
                    weight[idx] = math.log(1 / float(file.read()))

        weight = weight[0:ada_len]
        res = []
        for model_id in range(0, ada_len):
            res.append(predict(x_test, config, model_id))
        res = np.array(res).T
        for i in range(0, len(res)):
            avg = 0.0
            for j in range(0, ada_len):
                avg += (weight[j] * res[i][j])/np.sum(weight)
            y_predict.append(avg)
        # for i in range(0, len(res)):
        #     tmp = [0, 0, 0, 0, 0, 0] # [1][2][3][4][5] is useful
        #     for j in range(0, ada_len):
        #         tmp[res[i][j]] += weight[j]
        #     y_predict.append(np.argmax(tmp))
    elif config.ensemble_mode == 'SINGE':
        y_predict = predict(x_test, config)

    if not os.path.exists('result'):
        os.mkdir('result')
    with open('result/' + str(config) + "-result.csv", 'w') as file:
        file.write("id,predicted\n")
        for i in range(0, len(y_predict)):
            file.write(str(i + 1) + ',' + str(y_predict[i]) + '\n')
        file.close()