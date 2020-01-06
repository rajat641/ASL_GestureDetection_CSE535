from sklearn.ensemble import RandomForestClassifier
import pickle

# Function for Random_forest to predict labels using training and test data
def random_forest(train, test, trainLabels):
    # Declare variable to perform Linear SVM
    classifier = RandomForestClassifier(n_estimators=800, max_depth=2, random_state=3)

    filename = 'random_forest_model.sav'
    clf = pickle.load(open(filename, 'rb'))
    # clf = classifier.fit(train, trainLabels)
    # pickle.dump(clf, open(filename, 'wb'))
    prediction = clf.predict(test)

    return prediction
