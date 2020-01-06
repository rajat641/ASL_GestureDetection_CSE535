
from sklearn.linear_model import SGDClassifier
import pickle

def sgd_classifier(train, test, trainLabels):
    #declaring classifier
    classifier = SGDClassifier(loss='perceptron',random_state=5)
    
    filename = 'sgd_model.sav'
    # clf = classifier.fit(train, trainLabels)
    # pickle.dump(clf, open(filename, 'wb'))
    classifier = pickle.load(open(filename, 'rb'))
    # prediction = clf.predict(test)
    prediction = classifier.predict(test)
    return prediction
