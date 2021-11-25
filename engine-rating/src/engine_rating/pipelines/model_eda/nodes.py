import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
def missing_column_count(df):
    missing_feature_count = dict(df.drop('rating_engineTransmission',axis=1).isnull().sum())
    missing_feature_count = dict(sorted(missing_feature_count.items(), key=lambda x:x[1],reverse=True))
    print("Printing few missing value features")
    c=0
    for k,v in missing_feature_count.items():
        if v/len(df) > .99:
            print('column: {} , missing_count: {}'.format(k, v))
            c=c+1
    print("="*100)
    print("The number of features with 99% of its values missing are : ", c)
    return missing_feature_count

def drop_empty_features(df):
    df = df.drop('engineTransmission_engineOil_cc_value_9',axis=1)
    print("Shape of dataframe after droping features with all values missing :",df.shape)
    return df

def barplot(df): 
    fig, ax = plt.subplots()
    sns.barplot(x=df['rating_engineTransmission'].value_counts().index, y=df['rating_engineTransmission'].value_counts().values,ax=ax)
    plt.title('Frequency Distribution of Target Variable')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Engine Rating', fontsize=12)
    print(df['rating_engineTransmission'].value_counts())
    return fig

def remove_rating_1(df):
    df.drop(16022,inplace=True)
    df = df.reset_index(drop=True)
    print("Removed the datapoint with engine rating 1.0, since only one datapoint had rating 1.0")
    return df

def univariate_barplots(df,col1,col2):
    print("Univariate analysis on feature :",col1)
    df=df.fillna('Entry Missing')
    temp = df.groupby(col1)[col2]
    l = len(temp)
    fig,axes = plt.subplots(1,l,figsize=(40,10))
    c=0
    for i,j in temp:
        sns.barplot(x=j.value_counts(dropna=False).keys(),y=j.value_counts(dropna=False),ax=axes[c])
        axes[c].set_title(i)
        axes[c].set_ylabel('Number of Occurrences', fontsize=12)
        axes[c].set_xlabel('Engine Rating', fontsize=12)
        c+=1
    return fig

def univariate_analysis(df):
    print("Univariate analysis on some features with missing values greater than 90%")
    fig1=univariate_barplots(df,col1 = 'engineTransmission_clutch_cc_value_6',col2 = 'rating_engineTransmission')
    fig2=univariate_barplots(df,col1 = 'engineTransmission_engineOil_cc_value_8',col2 = 'rating_engineTransmission')
    fig3=univariate_barplots(df,col1 = 'engineTransmission_coolant_cc_value_1',col2 = 'rating_engineTransmission')

    print("Univariate analysis on some features with missing values less than 50%")
    fig4=univariate_barplots(df,col1 = 'engineTransmission_engineSound_cc_value_1',col2 = 'rating_engineTransmission')
    fig5=univariate_barplots(df,col1 = 'engineTransmission_engineSound_cc_value_0',col2 = 'rating_engineTransmission')
    fig6=univariate_barplots(df,col1 = 'engineTransmission_engineOil_cc_value_0',col2 = 'rating_engineTransmission')

    print("Univariate Analysis on some categorical features with category as [Yes or No].")
    fig7=univariate_barplots(df,col1 = 'engineTransmission_battery_value',col2 = 'rating_engineTransmission')
    fig8=univariate_barplots(df,col1 = 'engineTransmission_engineoilLevelDipstick_value',col2 = 'rating_engineTransmission')
    fig9=univariate_barplots(df,col1 = 'engineTransmission_engineOil',col2 = 'rating_engineTransmission')
    return fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def numerical_feature_processing(df):
    insp_dates = pd.DatetimeIndex(df['inspectionStartTime'])
    man_year = df['year']
    man_month = df['month']
    age = []
    for i in range(len(df)):
        age.append(diff_month(insp_dates[i],datetime(man_year[i],man_month[i],1)))
    df['Age'] = age
    df = df.drop(['inspectionStartTime','year','month'],axis=1)
    fig1 = numerical_feature_analysis(df,col1='odometer_reading',col2='rating_engineTransmission')
    fig2 = numerical_feature_analysis(df,col1='Age',col2='rating_engineTransmission')
    return df,fig1,fig2

def numerical_feature_analysis(df,col1,col2):
    fig, ax = plt.subplots()
    sns.kdeplot(x=col1, data=df, hue=col2, cumulative=False, common_norm=False)
    return fig
