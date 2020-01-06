from sklearn import svm
import pickle

# Function for SVM Linear Regression to predict labels using training and test data
def svm_linear(train, test, trainLabels):
    # Declare variable to perform Linear SVM
    classifier = svm.SVC(kernel='linear',gamma='scale')
    filename = 'svm_linear_model.sav'
    clf = pickle.load(open(filename, 'rb'))
    # clf = classifier.fit(train, trainLabels)
    # pickle.dump(clf, open(filename, 'wb'))
    prediction = clf.predict(test)
    
    return prediction
