import pandas as pd
import numpy as np
import pickle

train=pd.read_csv('train.csv')

pd.DataFrame()
print(train)
train["Item_Weight"] = train["Item_Weight"].fillna(train["Item_Weight"].mean())

#Random sample Imputation for Outlet_size feature
train['Random_Outlet_Size'] = train['Outlet_Size']
random_sample = train['Outlet_Size'].dropna().sample(train['Outlet_Size'].isnull().sum(),random_state=0)
random_sample.index =  train[train['Random_Outlet_Size'].isnull()].index
train.loc[train['Outlet_Size'].isnull(),'Random_Outlet_Size'] = random_sample

train.drop('Outlet_Size',axis=1,inplace=True)

#Handling categorical data
from sklearn.preprocessing import LabelEncoder
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
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state = 0)

#preproccessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

#Model creation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
model = RandomForestRegressor()
print("Entering model creation")
model.fit(xtrain,ytrain)
rypred = model.predict(xtest)
r2_score(ytest,rypred)
print("initial score:",r2_score)

#Hyperparameter tunning
#Hyperparamter tunning on randomforest
from sklearn.model_selection import RandomizedSearchCV
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




