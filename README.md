# ASL_Gesture_CSE535
An online Application Service that accepts Human Pose Skeletal key 
points of a sign video and return the label of the sign as a JSON Response. 
The Key points are generated using TensorFlow’s Pose Net.
A detailed version of the problem can be found here: [ASL_Detect_Problem.pdf](https://github.com/rajat641/CSE535-Mobile-FE/files/4026390/Assignment2_Thursday.9.pdf)

# CSE535Assignment2

## Group 30
- Reet Chatterjee
- Semira Chung
- Baani Khurana
- Rajat Singh

## Models
- SVM Linear Regression (Semira)
- Logistic Regression (Baani)
- Random Forest (Rajat)
- SGD Classifier (Reet)

## Built With
- Python
- Flask
- AWS EC2
- Nginx
- Gunicorn

## Files
- `app.py`
   - Uses Python Flask to run the app which is running on a server using an API call. Calls `main.py` to run the project.
- `main.py`
   - Goes through each and every training data containing JSON objects (book, car, gift, movie, sell, total) and parses them to train all four models.
   - For testing purposes and computing the accuracy, the training data is split into 80% training and 20% test.
   - In order to increase accuracy, unncessary columns are disregarded.
- `svm_linear.py`
   - A Python file that holds a function called `svm_linear(train, test, trainLabels)` which uses SVM Linear Regression to predict labels using training and test data.
   - `svm` is imported from a Python library called `sklearn` to perform SVM Linear Regression.
- `logistic_regression.py`
   - A Python file that holds a function called `logistic_regression(train, test, trainLabels)` which uses Logistic Regression to predict labels using training and test data.
   - `LogisticRegression` is imported from a Python library called `sklearn.linear_model` to perform Logistic Regression.
- `random_forest.py`
   - A Python file that holds a function called `random_forest(train, test, trainLabels)` which uses the Random Forest model to predict labels using training and test data.
   - `RandomForestClassifier` is imported from a Python library called `sklearn.ensemble` to perform the Random Forest functionality.
- `sgd_classifier.py`
   - A Python file that holds a function called `sgd_classifier(train, test, trainLabels)` which uses SGD Classifier to predict labels using training and test data.
   - `SGDClassifier` is imported from a Python library called `sklearn.linear_model` to perform the SGD Classifier.

*NOTE*: `svm_linear.py`, `logistic_regression.py`, `random_forest.py`, and `sgd_classifier.py` use `pickle` to save the models so that they can be used retrive information more efficiently and quickly instead of re-training the models everytime the application is run.

## API Endpoints

- `/` [Example: http://18.191.155.202/]
   - The Home page displaying the available Endpoints



- `/train/`   [Example: http://18.191.155.202/train/]
   - The Endpoint is used to call the the function to train four models on the training data. The function also save the models in the server for predictions.


- `/pred/`    [Example: http://18.191.155.202/pred/]

### Request in cURL
curl -X POST \
  http://18.191.155.202/pred/ \
  -H 'Content-Type: application/json' \
  -H 'cache-control: no-cache' \
  -d 'json text'

