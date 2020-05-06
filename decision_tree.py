from sklearn import tree
import joblib
import numpy as np
import os


def decision_tree_train(x_train, y_train, x_validation, y_validation, ensemble_mode: str, tree_id: int):
    model = tree.DecisionTreeClassifier(class_weight='balanced')

    model.fit(x_train, y_train)
    if not os.path.exists('model'):
        os.mkdir('model')

    if ensemble_mode == 'BAGGING':
        if not os.path.exists('model/' + ensemble_mode):
            os.mkdir('model/' + ensemble_mode)
        joblib.dump(model, 'model/' + ensemble_mode + '/dtree_' + str(tree_id) + '.pkl')
    elif ensemble_mode == 'ADA_BOOST_M1':
        pass
    elif ensemble_mode == "SINGLE":
        joblib.dump(model, 'model/dtree.pkl')
    else:
        print("unimplemented in decision_tree_train!")
        exit(0)

    print("current model accuracy on validating set: ", model.score(x_validation, y_validation))


def decision_tree_predict(words_data, ensemble_mode, model_id=''):
    if ensemble_mode == 'BAGGING':
        model = joblib.load('model/' + ensemble_mode + '/dtree_' + str(model_id) + '.pkl')
        result = model.predict(words_data)
    elif ensemble_mode == "ADA_BOOST_M1":
        model = joblib.load('model/' + ensemble_mode + '/dtree_' + str(model_id) + '.pkl')
        result = model.predict(words_data)
    elif ensemble_mode == 'SINGLE':
        model = joblib.load('model/tree_model.pkl')
        result = model.predict(words_data)
    else:
        print("unimplemented in decision_tree_predict!")
        exit(0)
    return result
