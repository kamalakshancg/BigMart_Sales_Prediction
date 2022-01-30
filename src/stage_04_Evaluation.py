import pandas as pd
import numpy as np
import joblib
import pickle
import os
import yaml
import argparse
from src.utils.all_utils import read_yaml,create_dir

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def Evaluation(config_path):

    contents = read_yaml(config_path)

    artifacts_dir = contents['artifacts']['artifacts_dir']
    raw_local_split_dir = contents['artifacts']['raw_local_split_dir']
   
    test_file_name = contents['artifacts']['local_test_file']
    raw_local_data_file_path = os.path.join(artifacts_dir,raw_local_split_dir)
    test_file_path = os.path.join(raw_local_data_file_path,test_file_name)


    train = pd.read_csv(test_file_path,index_col=0)

    print(train)
    train['Item_Weight'].fillna(train['Item_Weight'].mean(),inplace=True)

    #Random sample Imputation for Outlet_size feature
    train['Random_Outlet_Size'] = train['Outlet_Size']
    random_sample = train['Outlet_Size'].dropna().sample(train['Outlet_Size'].isnull().sum(),random_state=0)
    random_sample.index =  train[train['Random_Outlet_Size'].isnull()].index
    train.loc[train['Outlet_Size'].isnull(),'Random_Outlet_Size'] = random_sample

    train.drop('Outlet_Size',axis=1,inplace=True)

    #Handling categorical data
   
    lebel_encoder =  LabelEncoder()

    #replacing ('low fat',LF) variables to 'Low Fat' and 'reg' to 'Regular' beacause they are same 
    train['Item_Fat_Content'].replace({'reg':'Regular','low fat':'Low Fat','LF':'Low Fat'},inplace=True)

    #Performing label encodinf on categorical data
    train['Item_Type'] = lebel_encoder.fit_transform(train['Item_Type'])
    train['Outlet_Type'] = lebel_encoder.fit_transform(train['Outlet_Type'])
    train['Random_Outlet_Size'] = lebel_encoder.fit_transform(train['Random_Outlet_Size'])
    train['Outlet_Location_Type'] = lebel_encoder.fit_transform(train['Outlet_Location_Type'])
    train['Item_Fat_Content'] = lebel_encoder.fit_transform(train['Item_Fat_Content'])

    #removing outliers
    Q1 = train['Item_Visibility'].quantile(0.25)
    Q3 =  train['Item_Visibility'].quantile(0.75)
    IQR =  Q3-Q1
    lower_range = Q1-(1.5*IQR)
    upper_range = Q3+(1.5*IQR)
    train = train[(train['Item_Visibility']>lower_range) & (train['Item_Visibility']<upper_range)]

    #Feature Selection
    #we will remove Item_Identifier,Outlet_Identifier,Outlet_Establishment_Year because these features are not needed to predict the sales
    train.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1,inplace=True)

    #Splitting the data
    x = train.drop('Item_Outlet_Sales',axis=1)
    y = train['Item_Outlet_Sales']
    
    print("=======================================")
    print(x)
    #preproccessing
    scaler = StandardScaler()
    xtest = scaler.fit_transform(x)
    
    local_model_dir = contents['artifacts']['models']
    local_model_file = contents['artifacts']['model_file']
    raw_local_model_dir_path = os.path.join(artifacts_dir,local_model_dir)
    raw_local_model_file_path = os.path.join(raw_local_model_dir_path,local_model_file)

   
    best_r_model = joblib.load(raw_local_model_file_path)
     
    random_ypred = best_r_model.predict(xtest)
    print("r2_score: ",r2_score(y,random_ypred))

   
if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    #ModelCreation(config_path=parsed_args.config)
    Evaluation(config_path=parsed_args.config)