#Importing libraries 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Importing warnings
import warnings
warnings.filterwarnings('ignore')

#Importing the dataset
df = pd.read_excel("C:/Users/Dell/Desktop/Yash/Data Science/CarPrice_Assignment.xlsx", sheet_name = 'CarPrice_Assignment')

# Preprocessing 
df_numeric = df.select_dtypes(include = ['float64','int64']) #Only numeric datatypes 
df_numeric = df_numeric.drop(['car_ID'], axis = 1) #Car_Id is not a relevant feature. 
df['CarName'] = df['CarName'].apply(lambda x: x.strip())
car_name = df['CarName'].apply(lambda x: x.split(" ")[0]).to_frame()
df['car_company'] = car_name

df['car_company'].astype('category').value_counts() 

#Cleaning the dataset for name of cars
df.loc[(df['car_company'] == 'vw') | (df['car_company'] == 'vokswagen'), "car_company"] = 'volkswagen'
df.loc[(df['car_company'] == 'porcshce'),"car_company"] = 'porsche'
df.loc[(df['car_company'] == 'toyouta'), "car_company"] = 'toyota'
df.loc[(df['car_company'] == 'Nissan'), "car_company"] = 'nissan'
df.loc[(df['car_company'] == 'maxda'), "car_company"] = 'mazda'

df['car_company'].astype('category').value_counts() 

#Now, the model of cars do not significantly impact the model.
df = df.drop('CarName', axis=1)

#Converting the numbers written in words to Number format. 
def num_map(x):
    return x.map({'two':2,'three':3,'four':4,'five':5,'six':6,'eight':8,'twelve':12})
df[['cylindernumber', 'doornumber']] = df[['cylindernumber', 'doornumber']].apply(num_map)

#Categorical Data
df_categorical = df.select_dtypes(['object'])

#Converting the categorical features into dummies for the model
df_dummies = pd.get_dummies(df_categorical, drop_first = True) #We have less variables to deal with when we do drop first.

#Dropping the columns that are categorical from main dataframe
df = df.drop(list(df_categorical.columns), axis = 1)

#Now adding the categorical variables which had been converted to numerical values. 
df = pd.concat([df,df_dummies], axis = 1)
df = df.drop(['car_ID'], axis = 1)

#Model Building and evaluation
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, train_size = 0.80, test_size = 0.20, random_state = 42)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#Following are numeric variables and will be scaled
col_list = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'doornumber', 'cylindernumber', 'price']
df_train[col_list] = sc.fit_transform(df_train[col_list])

y_train = df_train.pop('price')
X_train = df_train

#Building the model
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)

#Using Recursive Function Elimination (RFE) to select most relevant features instead of using all 59 features
from sklearn.feature_selection import RFE
LinReg = LinearRegression()
rfe_1 = RFE(LinReg, 15)
rfe_1.fit(X_train, y_train)

#Model Evaluation
import statsmodels.api as sm
relevant_col = X_train.columns[rfe_1.support_]

#Subsetting training data for 15 selected columns
X_train_rfe1 = X_train[relevant_col]

#Adding a constant to the model
X_train_rfe1 = sm.add_constant(X_train_rfe1)

#Fitting model with 15 variables
LinReg_1 = sm.OLS(y_train, X_train_rfe1).fit()

#Checking the VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train_rfe1.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe1.values, i) for i in range(X_train_rfe1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)

#As VIF for few features is inf, we redo the model
rfe_2 = RFE(LinReg, 10)
rfe_2.fit(X_train, y_train)
relevant_col = X_train.columns[rfe_2.support_]
X_train_rfe2 = X_train[relevant_col]
X_train_rfe2 = sm.add_constant(X_train_rfe2)
LinReg_2 = sm.OLS(y_train, X_train_rfe2).fit()

#Check summary to see the ompact of the reducing the number of features. 
#Check if P > |t| is in controllable range. 
# Checking the VIF again. 
vif = pd.DataFrame()
vif['Features'] = X_train_rfe2.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe2.values, i) for i in range(X_train_rfe2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)

#We have to drop the variables which have very high VIF
X_train_rfe2.drop('car_company_subaru', axis = 1, inplace = True)

# Refitting with 9 variables
X_train_rfe2 = sm.add_constant(X_train_rfe2)

# Fitting the model with 9 variables
LinReg_2 = sm.OLS(y_train, X_train_rfe2).fit()

#Checking the VIF again. 
vif = pd.DataFrame()
vif['Features'] = X_train_rfe2.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe2.values, i) for i in range(X_train_rfe2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)

#After performing this, checking summary, a variable has gone P > 0.05. Drop it. 
X_train_rfe2.drop('enginetype_ohcf', axis = 1, inplace = True)

# Refitting with 8 variables
X_train_rfe2 = sm.add_constant(X_train_rfe2)

# Fitting the model with 8 variables
LinReg_2 = sm.OLS(y_train, X_train_rfe2).fit()

#New VIF for this model. 
vif = pd.DataFrame()
vif['Features'] = X_train_rfe2.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe2.values, i) for i in range(X_train_rfe2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)

#Making Predictions
df_test[col_list] = sc.transform(df_test[col_list])
y_test = df_test.pop('price')
X_test = df_test

#Finding the most relevant variables and testing on them
X_test_rfe2 = X_test[relevant_col]

# Let's now drop the variables we had manually eliminated as well
X_test_rfe2 = X_test_rfe2.drop(['enginetype_ohcf', 'car_company_subaru'], axis = 1)
X_test_rfe2 = sm.add_constant(X_test_rfe2)

##Using the LinReg model for preds now
y_pred = LinReg_2.predict(X_test_rfe2)

#Calculating the accuracy
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)