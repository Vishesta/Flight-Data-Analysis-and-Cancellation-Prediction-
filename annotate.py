import os
import settings
import pandas as pd

def read():
    acquisition = pd.read_csv(os.path.join(settings.PROCESSED_DIR, "processed.txt"))
    return acquisition

def predict_reason(acquisition):
    '''process the prediction data so that it is binary.  REASON (A,B,C or D) is chosen in 
       settings. This function sets the predictor to be just one of the for reasons.  To 
       predict any of the 4 reasons (ie CANCELLED = 1) use predict_cancel() function'''

    # fillna values with 0
    y = acquisition[['CANCELLATION_REASON']].fillna(value=0)
    # create dummies for each of the reasons A,B,C,D and 0, but use only the reason column
    y = pd.get_dummies(y['CANCELLATION_REASON'],prefix="REASON")[['REASON_'+settings.reason]]
    return y

def predict_cancel(acquisition):
    y = acquisition[['CANCELLED']]
    return y

def preprocess(acquisition, y):

    # features matrix. list of featuers set in settings.py
    acquisition = acquisition[settings.FEATURES]
    # combine with y, to drop NaN values
    acquisition = pd.concat([acquisition,y],axis=1)
    acquisition = acquisition.dropna()
   
    # create dummy features for categorical features, save new feature names to a new list
    new_dummy_features = []
    for feature in settings.dummy_features:
        dummy_df = pd.get_dummies(acquisition[feature],prefix=feature)
        new_dummy_features.append(list(dummy_df.columns))
        acquisition = pd.concat([acquisition,dummy_df], axis=1).drop([feature],axis=1)

    #flatten list of new dummy feature names
    new_dummy_features = [item for sublist in new_dummy_features for item in sublist]     

    return acquisition,new_dummy_features

def write(acquisition,new_dummy_features):
    acquisition.to_csv(os.path.join(settings.PROCESSED_DIR, "train.csv"), index=False)
    with open(os.path.join(settings.PROCESSED_DIR, "new_settings.txt"), "w") as fh:
        for item in new_dummy_features:
            fh.write("%s\n" % item)

if __name__ == "__main__":
    acquisition = read()
    if settings.PREDICTOR == ["CANCELLED"]:
        y = predict_cancel(acquisition)
    else:
        y = predict_reason(acquisition)
    acquisition, new_dummy_features = preprocess(acquisition, y)
    write(acquisition,new_dummy_features)
