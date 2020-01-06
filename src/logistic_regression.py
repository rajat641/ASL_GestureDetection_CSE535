from sklearn.linear_model import LogisticRegression
import  pickle

# Function for Logistic Regression to predict labels using training and test data
def logistic_regression(train, test, trainLabels):
    # Declare logistic regression classifier
    classifier = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    
    filename = 'logistic_regression_model.sav'
    clf = pickle.load(open(filename, 'rb'))
    # clf = classifier.fit(train, trainLabels)
    # pickle.dump(clf, open(filename, 'wb'))
    prediction = clf.predict(test)
    
    return prediction
