from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
def missing_feature_cols(df):
    missing_feature_count = dict(df.drop('rating_engineTransmission',axis=1).isnull().sum())
    missing_feature_count = dict(sorted(missing_feature_count.items(), key=lambda x:x[1],reverse=True))
    missing_feature_cols = []
    for k,v in missing_feature_count.items():
        if v/len(df) != 0.0:
            missing_feature_cols.append(k)
    print("="*100)
    print("The number of features that has missing values  : ", len(missing_feature_cols))
    fe_cols=[]
    for col in missing_feature_cols:
        df[col+'_fe']= df[col].isnull().astype(int)
        fe_cols.append(col+'_fe')
    feature_cols = list(df.columns)
    feature_cols = set(feature_cols) ^ set(fe_cols)
    feature_cols.remove('rating_engineTransmission')
    df = df.fillna('Null')
    return df,feature_cols,fe_cols

def train_cv_test_split(df):
    y = df['rating_engineTransmission']
    X = df.drop('rating_engineTransmission',axis=1)
    print(X.shape)
    print(y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)
    
    y_train = y_train*10
    y_train = y_train.astype(int)
    y_cv = y_cv*10
    y_cv = y_cv.astype(int)
    y_test = y_test*10
    y_test = y_test.astype(int)

    print(X_train.shape, y_train.shape)
    print(X_cv.shape, y_cv.shape)
    print(X_test.shape, y_test.shape)

    return X_train,y_train,X_cv,y_cv,X_test,y_test

def fit_response_encoding(X_train_feature,y_train):
  classes = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
  unique = np.unique(X_train_feature)
  l = np.array([X_train_feature,y_train])
  df = pd.DataFrame(l.T)
  d={}
  count = df.pivot_table(index =[0,1],aggfunc='size') 
  #print(count)
  for i in unique:
    res_enc=[]
    denom = np.count_nonzero(X_train_feature==i)
    #print("denom :",denom)
    for target in classes:
      if target in count[i].keys():
        res_enc.append((count[i][target])/denom)
      else:
        res_enc.append(0)    
    d[i] = res_enc
  return d

def transfrom_response_encoding(X_transform,d):
  encode=[]
  for i in X_transform:
    if i in d:
      encode.append(d[i])
    else:
      encode.append([0.5]*9)
  return np.array(encode)

def response_encoding(feature_cols,X_train,X_cv,X_test,y_train,y_cv,y_test):
    feature_cols.remove('Age')
    feature_cols.remove('odometer_reading')
    encoded_features=[]
    for feature in feature_cols:
        d = fit_response_encoding(X_train[feature],y_train)
        X_train_feature = transfrom_response_encoding(X_train[feature],d)
        X_cv_feature = transfrom_response_encoding(X_cv[feature],d)
        X_test_feature = transfrom_response_encoding(X_test[feature],d)
        encoded_features.append((X_train_feature,X_cv_feature,X_test_feature))

        print("After Vectorization, {} dimension changed to :".format(feature))
        print("="*50)
        print(X_train_feature.shape)
        print(X_cv_feature.shape)
        print(X_test_feature.shape)
        print("="*50)
    return encoded_features

def encoding_numerical_features(X_train,X_cv,X_test,y_train,y_cv,y_test):
    scaler = StandardScaler()
    scaler.fit(X_train['odometer_reading'].values.reshape(-1,1))
    X_train_odo = scaler.fit_transform(X_train['odometer_reading'].values.reshape(-1,1))
    X_cv_odo = scaler.fit_transform(X_cv['odometer_reading'].values.reshape(-1,1))
    X_test_odo = scaler.fit_transform(X_test['odometer_reading'].values.reshape(-1,1))

    scaler = StandardScaler()
    scaler.fit(X_train['Age'].values.reshape(-1,1))
    X_train_time = scaler.fit_transform(X_train['Age'].values.reshape(-1,1))
    X_cv_time = scaler.fit_transform(X_cv['Age'].values.reshape(-1,1))
    X_test_time = scaler.fit_transform(X_test['Age'].values.reshape(-1,1))

    return X_train_odo,X_cv_odo,X_test_odo,X_train_time,X_cv_time,X_test_time

def combining_features(encoded_features,X_train_odo,X_cv_odo,X_test_odo,X_train_time,X_cv_time,X_test_time,X_train,X_cv,X_test,fe_cols):
    #For Response Encoding
    X_tr=encoded_features[0][0]
    X_cr=encoded_features[0][1]
    X_te=encoded_features[0][2]
    for feature in encoded_features[1:]: 
        X_tr = np.hstack((X_tr,feature[0]))
        X_cr = np.hstack((X_cr,feature[1]))
        X_te = np.hstack((X_te,feature[2]))

    X_tr = np.hstack((X_tr,X_train_odo,X_train_time))
    X_cr = np.hstack((X_cr,X_cv_odo,X_cv_time))
    X_te = np.hstack((X_te,X_test_odo,X_test_time))

    X_tr = np.hstack((X_tr,X_train[fe_cols]))
    X_cr = np.hstack((X_cr,X_cv[fe_cols]))
    X_te = np.hstack((X_te,X_test[fe_cols]))
    print(X_tr.shape)
    print(X_cr.shape)
    print(X_te.shape)

    return X_tr,X_cr,X_te



