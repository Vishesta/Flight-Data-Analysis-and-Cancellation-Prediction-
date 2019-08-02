
# coding: utf-8

# FLIGHT DELAY ANALYSIS- Predict whether or not a flight will be cancelled

# In[30]:


# Imports

# local settings and specifications
import settings
import os
# pandas
import pandas as pd

# visual aids
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# machine learning
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ## Load and preview flight data 

# In[31]:



df1 = pd.read_csv(os.path.join(settings.DATA_DIR, "airlines.csv"))
df1.head()


# In[32]:


df2 = pd.read_csv(os.path.join(settings.DATA_DIR,"airports.csv"))
df2.head()


# In[33]:


df3 = pd.read_csv(os.path.join(settings.DATA_DIR,"Flights.csv"))
df3.head()


# In[34]:


df3.info()


# In[35]:


#How many flights were cancelled in the year 2015 ?

print(df3['CANCELLED'].value_counts())
sns.countplot(x='CANCELLED', data=df3)


# In[36]:


#Reason for cancellation -
sns.countplot(x="CANCELLATION_REASON",data=df3)


# In[37]:


#Since weather is the reason of highest cancellations ,we plot in which month did weather related cancellations happen-
sns.countplot(x = "MONTH",data=df3[df3["CANCELLATION_REASON"]=='B'])


# In[38]:


# next we will integrate the latitude/longitude information into the flights dataframe
df2 = df2.set_index("IATA_CODE")
for loc in ['ORIGIN','DESTINATION']:
        for dir in ['LATITUDE','LONGITUDE']:
            df3[loc+'_'+dir] =(df2[dir][df3[loc+"_AIRPORT"][:]]).reset_index()[[dir]]
df3.head()


# In[39]:


def barplot_feature(df,feature, target = 'CANCELLED',cut = False, bins = 3):
    '''Plot of cancellation rate for various features. More continuous features can be binned together to convert
    to ranges of values
    
    df: the dataframe with the flight data
    feature: the feature to be plotted, must be a string a match a feature in df
    target: the prediction value, must be a string and match a feature in df
    cut: use cut if feature has continuous values.  '''
    if cut == True:
        # define a new feature which specifies a range of values for 'feature'.  ranges are determined according to 'bin'
        df['bins'] = pd.cut(df3[feature],bins)
        group_df = df[['bins',target]].groupby(['bins']).mean()
        return group_df.plot(kind='bar')
        #plt.title(feature)
    else:
        group_df = df[[feature,target]].groupby([feature],as_index=False).mean()
        return sns.barplot(x = feature, y = target, data=group_df)
        


# In[40]:


#how much does airport location play a role?
barplot_feature(df3,'DESTINATION_LATITUDE',cut = True, bins = 10)
plt.title('Destination_Latitude')
barplot_feature(df3,'ORIGIN_LATITUDE',cut = True, bins = 10)
plt.title('Origin_Latitude')
barplot_feature(df3,'DESTINATION_LONGITUDE',cut = True, bins = 10)
plt.title('Destination_Longitude')
barplot_feature(df3,'ORIGIN_LONGITUDE',cut = True, bins = 10)
plt.title('Origin_Longitude')


# In[41]:


#day of week with highest cancellation
barplot_feature(df3,'DAY_OF_WEEK')


# In[42]:


#airline analysis
barplot_feature(df3,'AIRLINE')


# In[43]:



def preprocess(dataframe,features,dummy_features, predictor, reason='B'):
    
    if predictor == ["CANCELLED"]:
        y = dataframe[predictor]
    else:
       
        y = dataframe[['CANCELLATION_REASON']].fillna(value=0)
      
        y = pd.get_dummies(y['CANCELLATION_REASON'],prefix="REASON")[['REASON_' + reason]]
    
   
    processed = dataframe[features]
    
    processed = pd.concat([processed,y],axis=1)
    processed = processed.dropna()
   
   
    new_dummy_features = []
    for feature in dummy_features:
        dummy_df = pd.get_dummies(processed[feature],prefix=feature)
        new_dummy_features.append(list(dummy_df.columns))
        processed = pd.concat([processed,dummy_df], axis=1).drop([feature],axis=1)

    
    new_dummy_features = [item for sublist in new_dummy_features for item in sublist]     

    return processed,new_dummy_features


# In[44]:



features = ["MONTH","DAY_OF_WEEK","AIRLINE","SCHEDULED_DEPARTURE","SCHEDULED_ARRIVAL",
                       "DISTANCE", "SCHEDULED_TIME","ORIGIN_LATITUDE","ORIGIN_LONGITUDE",
                       "DESTINATION_LATITUDE","DESTINATION_LONGITUDE"]
dummy_features = ["MONTH","DAY_OF_WEEK","AIRLINE"]
predictor = ["CANCELLED"]

processed,new_dummy_features = preprocess(df3,features,dummy_features,predictor)
processed.head()


# In[45]:



def compute_false_negatives(target, predictions):
    
    df = pd.DataFrame({"target": target, "predictions": predictions})
    return df[(df["target"] == 1) & (df["predictions"] == 0)].shape[0] / (df[(df["target"] == 1)].shape[0] + 1)

def compute_false_positives(target, predictions):
   
    df = pd.DataFrame({"target": target, "predictions": predictions})
    return df[(df["target"] == 0) & (df["predictions"] == 1)].shape[0] / (df[(df["target"] == 0)].shape[0] + 1)


# In[46]:



def split(processed, predictor, reason="B"):
    #prepare predictor, feature data, with correct shape
    if predictor == ["CANCELLED"]:
        y = processed[predictor].values
        X = processed.drop(predictor,axis=1)
    else:
        y = processed[['REASON_'+reason]].values
        X = processed.drop(['REASON_'+reason],axis=1)
    c,r = y.shape
    y = y.reshape(c,)
    #split into train and test sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    return X_train,X_test,y_train,y_test

def scale(X_train,X_test,new_dummy_features):
    X_train_dummy = X_train[new_dummy_features].values
    X_test_dummy = X_test[new_dummy_features].values
    X_train = X_train.drop(new_dummy_features,axis=1).values
    X_test = X_test.drop(new_dummy_features,axis=1).values
    #standarize non dummy features using training set
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    #recombine training sets
    X_train_final = np.concatenate((X_train_transformed,X_train_dummy),axis=1)
    #scale test set using training fits
    X_test_transformed = scaler.transform(X_test)
    #recombine
    X_test_final = np.concatenate((X_test_transformed,X_test_dummy),axis=1)
    return X_train_final, X_test_final

def predict_score(X_train_final,y_train,X_test_final,y_test):
    # train a logistic regression model
    clf = LogisticRegression(random_state=1, class_weight="balanced").fit(X_train_final, y_train)
    predictions = clf.predict(X_test_final)
    score = clf.score(X_test_final,y_test)
    return predictions, score


# In[47]:


#If the flight has been cancelled or not
X_train,X_test,y_train,y_test = split(processed,predictor)
X_train_final,X_test_final = scale(X_train,X_test,new_dummy_features)
predictions,score = predict_score(X_train_final,y_train,X_test_final,y_test)
fn = compute_false_negatives(y_test, predictions)
fp = compute_false_positives(y_test, predictions)
print("Accuracy Score: {}".format(score))
print("False Negatives: {}".format(fn))
print("False Positives: {}".format(fp))


# ## Predict weather related cancellations
# 

# In[48]:



predictor = ["CANCELLATION_REASON"]
processed,new_dummy_features = preprocess(df3,features,dummy_features,predictor)
X_train,X_test,y_train,y_test = split(processed,predictor)
X_train_final,X_test_final = scale(X_train,X_test,new_dummy_features)
predictions,score = predict_score(X_train_final,y_train,X_test_final,y_test)
fn = compute_false_negatives(y_test, predictions)
fp = compute_false_positives(y_test, predictions)
print("Accuracy Score: {}".format(score))
print("False Negatives: {}".format(fn))
print("False Positives: {}".format(fp))


# In[49]:


#RandomForestClassifier
model= RandomForestClassifier()
model.fit(X_train_final, y_train)
predicted= model.predict(X_test_final)
print("Random Forest Classifier: {}".format(predicted))
score = model.score(X_test_final,y_test)
print("Random Forest Classifier: {}".format(score))


# In[50]:


#K-means
k_means = KMeans(n_clusters=3)
k_means = k_means.fit(X_train_final,y_test)
labels=k_means.predict(X_test_final)
centroids=k_means.cluster_centers_

print("Kmeans Classifier :" , labels)


# In[51]:



