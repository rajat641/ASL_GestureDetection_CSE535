

import numpy as np
import json
import os
from pandas.io.json import json_normalize
from sklearn.linear_model import SGDClassifier

import svm_linear, logistic_regression, random_forest , sgd_classifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# padding function that allows all arrays to be of equal size
def pad(type, l, content, width):
    # if type == "train":
    for i in l:
        i.extend([content] * (width - len(i)))
    # else:
    #     l.extend([content] * (width - len(l)))
    return l



def pred(data):
    # initialize variables
    folders = ['book', 'car', 'gift', 'movie', 'sell', 'total']
    train = []
    test = []
    allLens = []
    # with open(file_path) as data_file:
    #     data = json.load(data_file)
    df = json_normalize(data, 'keypoints', ['score'], record_prefix='keypoints_')  # nested JSON to Data Frame
    df = df.reindex(columns=['score', 'keypoints_score', 'keypoints_position.x', 'keypoints_position.y'])
    df = np.array(df)
    df = np.reshape(df, (-1, 68)) # reshape the data so that each sub-array is one video frame
    df = df[:, 12:44] # select ears to the wrist data as features
    df = list(df.ravel())  # convert Data Frame to list and make it 1D
    test.append(df)  # add each file's data frame to allFiles
    allLens.append(len(df))
    # maxLen = max(allLens)
    test = pad("test", test, 0, 10368)
    # print(len(test[0]))
    # print(test)

    import pickle
    ## Loading models from .sav files
    loaded_model1 = pickle.load(open('./sgd_model.sav', 'rb'))
    pred_sgd = loaded_model1.predict(test)
    loaded_model2 = pickle.load(open('./svm_linear_model.sav', 'rb'))
    loaded_model3 = pickle.load(open('./random_forest_model.sav', 'rb'))
    loaded_model4 = pickle.load(open('./logistic_regression_model.sav', 'rb'))
    pred_svm = loaded_model2.predict(test)
    pred_random = loaded_model3.predict(test)
    pred_logictic = loaded_model4.predict(test)



    data = {}
    data['1'] = pred_svm[0]
    data['2'] = pred_logictic[0]
    data['3'] = pred_random[0]
    data['4'] = pred_sgd[0]
    json_data = json.dumps(data, indent=3)
    print(json_data)
    return json_data

# pred()
