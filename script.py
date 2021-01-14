#########################################   IMPORTS   #####################################################################
import pandas as pd
import numpy as np
import pickle
from flask import Flask, flash, redirect, render_template, request, session, abort
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)

#########################################   READING DATASET   #############################################################

df = pd.read_csv('cleveland.csv', header = None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

#########################################   DEALING WITH EMPTY VALUES  ####################################################

### 1 = male, 0 = female
df.isnull().sum()

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['sex'] = df.sex.map({0: 'female', 1: 'male'})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

#########################################  VISUALISATION OF DATASET   #####################################################

import matplotlib.pyplot as plt
import seaborn as sns

# distribution of target vs age
sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20})
sns.catplot(kind = 'count', data = df, x = 'age', hue = 'target', order = df['age'].sort_values().unique())
plt.title('Variation of Age for each target class')
plt.show()


# barplot of age vs sex with hue = target
sns.catplot(kind = 'bar', data = df, y = 'age', x = 'sex', hue = 'target')
plt.title('Distribution of age vs sex with the target class')
plt.show()

df['sex'] = df.sex.map({'female': 0, 'male': 1})


################################## PREPROCESSING ##########################################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#print(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#########################################   SVM   #########################################################################
from sklearn.svm import SVC
SVM_classifier = SVC(kernel = 'rbf',probability=True)
SVM_classifier.fit(X_train, y_train)

filename = 'Pickled_SVM_model.sav'
pickle.dump(SVM_classifier, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_modelSVM = pickle.load(open(filename, 'rb'))
#resultsvm = loaded_model.score(X_test, Y_test)

# Predicting the Test set results
y_pred = SVM_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = SVM_classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

train_accuracy_svm = (cm_train[0][0] + cm_train[1][1])/len(y_train)
test_accuracy_svm = (cm_test[0][0] + cm_test[1][1])/len(y_test)
print()
#print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
#print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


#########################################   Naive Bayes  #################################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)

filename = 'Pickled_NB_model.sav'
pickle.dump(NB_classifier, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_modelNB = pickle.load(open(filename, 'rb'))
# Predicting the Test set results
y_pred = NB_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = NB_classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

train_accuracy_nb = (cm_train[0][0] + cm_train[1][1])/len(y_train)
test_accuracy_nb = (cm_test[0][0] + cm_test[1][1])/len(y_test)
print()
#print('Accuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
#print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


#########################################   Logistic Regression  ##########################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
Regression_classifier = LogisticRegression()
Regression_classifier.fit(X_train, y_train)
filename = 'Pickled_LogRegr_model.sav'
pickle.dump(Regression_classifier, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model_logRegr = pickle.load(open(filename, 'rb'))
# Predicting the Test set results
y_pred = Regression_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = Regression_classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

train_accuracy_regression = (cm_train[0][0] + cm_train[1][1])/len(y_train)
test_accuracy_regression = (cm_test[0][0] + cm_test[1][1])/len(y_test)

print()
#print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
#print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

# importing Flask and other modules
from flask import Flask, request, render_template

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function

@app.route('/', methods =["GET", "POST"])
def predictDetailedFunction():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       Age = eval(request.form.get("age"))
       # getting input with name = lname in HTML form
       gender = eval(request.form.get("gender"))
       cp = eval(request.form.get("Chestpain"))
       bp = eval(request.form.get("BloodPressure"))
       choles =eval(request.form.get("Cholestrol"))
       blood_sugar = eval(request.form.get("blood_sugar"))
       ecg = eval(request.form.get("ecg"))
       heart_rate = eval(request.form.get("heart_rate"))
       exercise_angina = eval(request.form.get("exercise_angina"))
       old_peak = eval(request.form.get("old_peak"))
       slopes = eval(request.form.get("slopes"))
       vessels = eval(request.form.get("vessels"))
       thala = eval(request.form.get("thala"))

       array = [Age,gender,cp,bp,choles,blood_sugar,ecg,heart_rate,exercise_angina,old_peak,slopes,vessels,thala]
       #print(array)
       if request.form['final']== 'Svm':
           result = loaded_modelSVM.predict([array])
           confidence = loaded_modelSVM.predict_proba([array])
           modelAccuracy = test_accuracy_svm
           print(loaded_modelSVM.predict_proba([array]))
       elif request.form['final'] == 'Naive':
           result = loaded_modelNB.predict([array])
           modelAccuracy = test_accuracy_nb
           confidence = loaded_modelNB.predict_proba([array])
           print(loaded_modelNB.predict_proba([array]))
       else:
           result = loaded_model_logRegr.predict([array])
           modelAccuracy = test_accuracy_regression
           confidence = loaded_model_logRegr.predict_proba([array])
           print(loaded_model_logRegr.predict_proba([array]))
       if result == 1:
           message = "You don't have a heart disease.";
       else:
           message = "Chances are you might have heart disease."
       accu = int(modelAccuracy*100000)
       accu = accu/1000
       print(accu)

       confidence = int(confidence[0][1]*100000)

       confidence = confidence/1000
       return render_template("result.html",msg = message, accuracy = accu, conf = confidence)

       #return "Your name is "+Age+" " + gender + " " + cp
    return render_template("heart_form_detailed.html") # for the first time this will open

if __name__=='__main__':
   app.run(debug=True)
