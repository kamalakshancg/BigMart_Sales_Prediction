import pandas as pd
import numpy as np
import pickle
import joblib
import os
import yaml
import argparse
from src.utils.all_utils import read_yaml,create_dir
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def ModelCreation(config_path):


    
    contents = read_yaml(config_path)

    artifacts_dir = contents['artifacts']['artifacts_dir']
    raw_local_split_dir = contents['artifacts']['raw_local_split_dir']
   
    train_file_name = contents['artifacts']['local_train_file']
    raw_local_data_file_path = os.path.join(artifacts_dir,raw_local_split_dir)
    train_file_path = os.path.join(raw_local_data_file_path,train_file_name)




    train = pd.read_csv(train_file_path)

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
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state = 0)

    #preproccessing
   
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    #Model creation
   
    model = RandomForestRegressor()
    print("Entering model creation")
    model.fit(xtrain,ytrain)
    rypred = model.predict(xtest)
    r2_score(ytest,rypred)
    print("initial score:",r2_score)

    #Hyperparameter tunning
    #Hyperparamter tunning on randomforest
   
    param_grid ={
            'n_estimators':[i for i in range(200,2001,200)],
            'max_depth':[int(i) for i in np.linspace(10,1000,10)],
            'min_samples_split' : [2, 5, 10,14],
            'min_samples_leaf' : [1, 2, 4,6,8],
            'max_features' : ['auto', 'sqrt','log2']
    }
    best_r_model = RandomizedSearchCV(estimator=RandomForestRegressor(),param_distributions=param_grid,cv = 3)
    print(best_r_model.estimator)
    best_r_model.fit(xtrain,ytrain)
    random_ypred = best_r_model.predict(xtest)
    r2_score(ytest,random_ypred)
    print("Running hyperparameter tunning")
    print("r2_score: ",r2_score(ytest,random_ypred))

    local_model_dir = contents['artifacts']['models']
    local_model_file = contents['artifacts']['model_file']
    raw_local_model_dir_path = os.path.join(artifacts_dir,local_model_dir)
    raw_local_model_file_path = os.path.join(raw_local_model_dir_path,local_model_file)
    create_dir(dirs=[raw_local_model_dir_path])
    
    with open(raw_local_model_file_path,"wb") as model_file_point:
        pickle.dump(best_r_model,model_file_point)
        
    print("raw_local_model_file_path:",raw_local_model_file_path)
    joblib.dump(best_r_model,raw_local_model_file_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    #ModelCreation(config_path=parsed_args.config)
    ModelCreation(config_path=parsed_args.config)