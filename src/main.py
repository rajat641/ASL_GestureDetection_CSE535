import numpy as np
import json
import os
from pandas.io.json import json_normalize

import svm_linear, logistic_regression, random_forest , sgd_classifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# padding function that allows all arrays to be of equal size
def pad(type, l, content, width):
    if (type == True):
        for i in l:
            i.extend([content] * (width - len(i)))
    else:
        l.extend([content] * (width - len(l)))
    return l


def main():
    # initialize variables
    folders = ['book', 'car', 'gift', 'movie', 'sell', 'total']
    train = []
    test = []
    trainLabels = []
    testLabels = []
    allFiles = []
    allLens = []
    split_training_and_testing = False

    for folder in folders:
        files = os.listdir('./Thursday_Assignment_2_json/' + folder + "/")
        for file in files:
            # opens JSON object and stores it into variable called data
            with open('./Thursday_Assignment_2_json/' + folder + "/" + file) as data_file:
                data = json.load(data_file)

            df = json_normalize(data, 'keypoints', ['score'], record_prefix='keypoints_')  # nested JSON to Data Frame
            df = df.reindex(columns=['score', 'keypoints_score', 'keypoints_position.x', 'keypoints_position.y'])
            df = np.array(df)
            df = np.reshape(df, (-1, 68)) # reshape the data so that each sub-array is one video frame
            df = df[:, 12:44] # select ears to the wrist data as features
            df = list(df.ravel())  # convert Data Frame to list and make it 1D
            allFiles.append(df)  # add each file's data frame to allFiles
            allLens.append(len(df))  # keep track of each data frames' length to determine maximum
        
        if (split_training_and_testing == True):
            indexSplit = round(len(allFiles) * .8)  # determine index to split to create 80% 20% split
        else:
            indexSplit = round(len(allFiles) * 1)   # trains all the data (100%) and no test data
        
        train += allFiles[:indexSplit]  # training
        trainLabels += [folder] * indexSplit
        test += allFiles[indexSplit:]  # testing
        testLabels += [folder] * (len(allFiles) - indexSplit)

        allFiles = []  # empty allFiles for next folder

    maxLen = max(allLens)  # determine maximum length for padding
    train = pad(True, train, 0, maxLen)  # training data padded
    test = pad(True, test, 0, maxLen)

    # with 100% training data it uses this default test input into the ML models
    if (split_training_and_testing == False):
    # keypoints.json input as test
        with open('./keypoints.json') as input_file:
            input_data = json.load(input_file)
        input_df = json_normalize(input_data, 'keypoints', ['score'],
                                record_prefix='keypoints_')  # nested JSON to Data Frame
        input_df = input_df.reindex(columns=['score', 'keypoints_score', 'keypoints_position.x', 'keypoints_position.x'])
        input_df = list(np.array(input_df).ravel())  # convert Data Frame to list
        test = pad(False, input_df, 0, maxLen)  # testing data padded
        test = [test]

    # Training and Testing Functions
    svm_linear_prediction = svm_linear.svm_linear(train, test, trainLabels)
    logistic_regression_prediction = logistic_regression.logistic_regression(train, test, trainLabels)
    random_forest_prediction = random_forest.random_forest(train, test, trainLabels)
    sgd_classifier_prediction = sgd_classifier.sgd_classifier(train, test, trainLabels)

    # Prints testing preductions and accuracies if 80/20 training and test data
    if (split_training_and_testing == True):
        print("SVM Linear Accuracy: ", accuracy_score(testLabels, svm_linear_prediction))
        print("Logistic Regression Accuracy: ", accuracy_score(testLabels, logistic_regression_prediction))
        print("Random Forest Accuracy: ", accuracy_score(testLabels, random_forest_prediction))
        print("Ridge Classifier Accuracy: ", accuracy_score(testLabels, sgd_classifier_prediction))
        print("------------------------------------------")
        print("SVM Linear Predictions: " + str(svm_linear_prediction))
        print("Logistic Regression Predictions: " + str(logistic_regression_prediction))
        print("Random Forest Predictions: " + str(random_forest_prediction))
        print("Ridge Classifier Predictions: " + str(sgd_classifier_prediction))
        print("------------------------------------------")
        print("Testing Labels: " + str(testLabels))

        data = {}
        data['1'] = svm_linear_prediction[0]
        data['2'] = logistic_regression_prediction[0]
        data['3'] = random_forest_prediction[0]
        data['4'] = sgd_classifier_prediction[0]
        output = json.dumps(data, indent=3)
    else: # otherwise for 100% training data, it just trains
        output = "100% Training Successful"
    return output
