import pandas as pd
def cleaning(df):
    print("\nThe dataset has {} rows and {} columns".format(df.shape[0],df.shape[1]))
    print("\nThe class label distribution is as shown below\n")
    print(df['rating_engineTransmission'].value_counts())
    print("\nDropping feature appointment_id as it is just an index")
    df = df.drop('appointmentId',axis=1)
    col1 = df.columns

    print("\nRemoving duplicates ")
    df = df.drop_duplicates(keep='first')
    df = df.T.drop_duplicates().T
    col2 = df.columns

    print("{} is a duplicate column and is removed".format(set(col1) ^ set(col2)))
    print("Shape of the dataset after data cleaning :", df.shape)
    return df

