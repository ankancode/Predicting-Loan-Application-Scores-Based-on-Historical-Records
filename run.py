import pandas as pd
import numpy as np
from sklearn import preprocessing
from xgboost import XGBClassifier
import argparse
import pickle


def normalize_values(col, col_name):
    '''
    Performs Normalization

    Parameters:
    1) col: Input Column
    2) col_name: Column Name

    return transformed column
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(col)
    col_transformed = pd.DataFrame(x_scaled, columns=[col_name])
    return col_transformed


def normalize(data):
    '''
    Drop Columns which are not required and normalizes the required columns

    Parameters:
    1) data: Test Dataset given in the input

    returns data
    '''
    columns = ["Id", "Product_Info_2", "Medical_History_10",
               "Medical_History_32", "Medical_History_24",
               "Medical_History_15", "Family_Hist_5", "Family_Hist_3",
               "Family_Hist_2", "Insurance_History_5", "Family_Hist_4"]

    data.drop(columns, inplace=True, axis=1)

    for i in data.columns:
        if (data[i].dtypes == 'int64' or data[i].dtypes == 'float64' and
                i != 'Response'):
            try:
                col_split = i.split('_')[1]
                if col_split != 'Keyword':
                    col_transformed = normalize_values(
                        data[i].reshape(-1, 1), i)
                    data.drop(i, inplace=True, axis=1)
                    data = pd.concat([data, col_transformed], axis=1)
                else:
                    data[i] = data[i].astype('str')
            except Exception as e:
                pass
    return data


def main():
    '''
    Main function, calls all the neccesary modules
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="path to test data file")
    args = vars(ap.parse_args())
    data = pd.read_csv(args["data"])

    X_test = normalize(data)

    with open('final_model.pickle', 'rb') as handle:
        final_rgr = pickle.load(handle)

    prediction = final_rgr.predict(np.array(X_test))

    data_df = pd.DataFrame(data=prediction, columns=['prediction'])

    data_df.to_csv("prediction.csv", index=False)

if __name__ == "__main__":
    main()
    print(" DONE !!! ")
