import os
import settings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def read():
    train = pd.read_csv(os.path.join(settings.PROCESSED_DIR, "train.csv"))
    with open(os.path.join(settings.PROCESSED_DIR, "new_settings.txt"), "r") as f:
        new_dummy_features = f.read().splitlines() 
    return train, new_dummy_features

def get_y(train):
    if settings.PREDICTOR == ["CANCELLED"]:
        y = train[settings.PREDICTOR].values
    else:
        y = train[['REASON_'+settings.reason]].values
    c,r = y.shape
    y = y.reshape(c,)
    return y

def get_X(train,y):
    if settings.PREDICTOR == ["CANCELLED"]:
        X = train.drop(settings.PREDICTOR,axis=1)
    else:
        X = train.drop(['REASON_'+settings.reason],axis=1)
    return X

def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = settings.test_size,
                                                        random_state=1)
    return X_train,X_test,y_train,y_test

def train_model(X_train,X_test,y_train,y_test,new_dummy_features):
    X_train_dummy = X_train[new_dummy_features].values
    X_test_dummy = X_test[new_dummy_features].values
    X_train = X_train.drop(new_dummy_features,axis=1).values
    X_test = X_test.drop(new_dummy_features,axis=1).values
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_train_final = np.concatenate((X_train_transformed,X_train_dummy),axis=1)
    clf = LogisticRegression(random_state=1, class_weight="balanced").fit(X_train_final, y_train)
    X_test_transformed = scaler.transform(X_test)
    X_test_final = np.concatenate((X_test_transformed,X_test_dummy),axis=1)
    predictions = clf.predict(X_test_final)
    score = clf.score(X_test_final,y_test)
    return predictions, score


if __name__ == "__main__":
    train, new_dummy_features = read()
    y = get_y(train)
    X = get_X(train,y)
    X_train,X_test,y_train,y_test = split(X,y)
    predictions,score = train_model(X_train,X_test,y_train,y_test,new_dummy_features)
    print("Accuracy Scores Of Algorithms ")
  
    print("Logistic Regression: {}".format(score))
   




#Random Forest 
model= RandomForestClassifier()
model.fit(X, y)
predicted= model.predict(X_test)
score=model.score(X_test, y_test)
print("Random Forest Classifier: {}".format(predicted))
print("Random Forest Classifier: {}".format(score))


 #Kmeans classifier


k_means = KMeans(n_clusters=3, random_state=0)
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(X_test)
score=model.score(X_test, y_test)
print("K Means: {}".format(predicted))



