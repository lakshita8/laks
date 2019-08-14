## BOSTON HOUSE CASE STUDY ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import seaborn as sb
import statsmodels.api as sm
from sklearn import datasets

data = pd.read_csv("F:\\lakshita\\BSE mchine learning\\Linear Regression\\boston_data.csv")
print(data)
print(data.keys())
print(data.shape)
print(data.dtypes)
print(data.head())

plt.hist(data['medv'],bins=20)
#density plot and histogram by seaborn library
sb.distplot(data['medv'], hist= True , kde = True,bins=20 , color ='red'
            , hist_kws={'edgecolor': 'black'}, kde_kws= {'linewidth':4})
sb.pairplot(data,kind='reg')
## FOR  FINDING OUTLIER OUTLIER
plt.boxplot(data['medv'])
plt.boxplot(data['crim'])
plt.boxplot(data['zn'])

# finding quantile
# finding outlier without boxplot
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr=q3-q1
print(iqr)
p= ((data<(q1 - 1.5*iqr)))|((data> (q3+ 1.5*iqr )))
print(p)

# CORRECTING OUTLIER
UL= (q3 + 1.5*iqr)
print(UL)
mask = data.crim > 10.00932
data.loc[mask,'crim']=10.00932


mask = data.zn > 31.25000
data.loc[mask,'zn']=31.25000

mask = data.rm >  7.71900
data.loc[mask,'rm']= 7.71900

mask = data.dis > 9.92350
data.loc[mask,'dis']=9.92350

mask = data.lstat > 31.57250
data.loc[mask,'lstat']= 31.57250

mask = data.medv >  36.85000
data.loc[mask,'medv']= 36.85000

## missing value
 
pd.isna(data)

data.info()
data.isnull().sum()

#missing vaue treatment
#mean_value=data['zn'].mean()
#data['zn']=data['zn'].fillna(mean_value) #replace null fill mean value

# correlation
corr=data.corr()
print(corr)

# declare dependent variable and create dep and indep datasets
dep= "medv"
x = data.drop(dep,axis=1) #independent 
y = data[dep] #dependent
x=sm.add_constant(x)

#split data into train and test datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=5)
                                              #random _state is the seed used by the random number generator
# model building
lm=sm.OLS(y_train , x_train).fit()
lm.summary()   

lm1 = sm.OLS(y_train, x_train.drop(['zn','indus','indus','chas','age'],
                                  axis=1)).fit()
lm1.summary()             
                            

lm2 = sm.OLS(y_train, x_train.drop(['zn','indus','indus','chas','age','nox','crim','rm','rad','tax','lstat','ptratio'],
                                  axis=1)).fit()
lm2.summary()             
             

from statsmodels.stats.outliers_influence import variance_inflation_factor
x_train=x_train.drop(['zn','indus','chas','nox','crim','rm','rad','age','tax','lstat'], axis=1)
[variance_inflation_factor(x_train.values, j) for j in range(x_train.shape[1])]

#Prediction
pred_test=lm2.predict(x_test.drop(['zn','indus','chas','nox','crim','rm','rad','age','tax','lstat','ptratio'],axis=1))
err_test=np.abs(y_test - pred_test)
print(err_test)

#MAPE
import numpy as np

def mean_absolute_percentage_error(y_test, pred_test): 
    y_test, pred_test = np.array(y_test), np.array(pred_test)
    return np.mean(np.abs((y_test - pred_test) / y_test)) * 100

mean_absolute_percentage_error(y_test, pred_test)

#Linearity
#The Null hypothesis is that the regression is correctly modeled as linear
sm.stats.diagnostic.linear_harvey_collier(lm2)
############################################################3

# since the mean absolute error is high now we have to do the transformation
#by taking log of y
# Log Transform
data['logmedv']=np.log(data['medv'])


#IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
p=((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR)))

#Outliers Treatment
UL=(Q3 + 1.5 * IQR)

mask = data.crim > 10.00932
data.loc[mask,'crim'] = 10.00932

mask = data.zn > 31.25000
data.loc[mask,'zn'] = 31.25000

mask = data.rm >  7.71900
data.loc[mask,'rm'] =  7.71900

mask = data.dis >  9.92350
data.loc[mask,'dis'] =  9.92350

mask = data.lstat > 31.57250
data.loc[mask,'lstat'] =  31.57250

mask = data.logmedv > 3.788572
data.loc[mask,'logmedv'] =  3.788572

#Correlation
corr=data.corr()


# Declare Dependent variable & create independent & dependent datasets
dep = "logmedv"
X = data.drop([dep,"medv"],axis=1)
Y = data[dep]

# with statsmodels
X = sm.add_constant(X) # adding a constant


#Split data into train & test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,
        Y,test_size=0.2,random_state=5) #random_state is the seed used by the random number generator;


##  MODEL BUILDING
lm=sm.OLS(Y_train, X_train).fit()
lm.summary()


lm1=sm.OLS(Y_train, X_train.drop(['zn','indus','chas','age'], 
                                 axis=1)).fit()
lm1.summary()


lm2=sm.OLS(Y_train, X_train.drop(['zn','indus','chas','nox','crim',
                                  'rm','rad','age','tax','lstat',
                                  'ptratio'], axis=1)).fit()
lm2.summary()


from statsmodels.stats.outliers_influence import variance_inflation_factor
x_train=X_train.drop(['zn','indus','chas','nox','crim','rm','rad','age','tax','lstat'], axis=1)
[variance_inflation_factor(x_train.values, j) for j in range(x_train.shape[1])]



#Prediction
pred_test=lm2.predict(X_test.drop(['zn','indus','chas','nox','crim','rm','rad','age','tax','lstat','ptratio'],axis=1))
err_test=np.abs(Y_test - pred_test)
print(err_test)


def mean_absolute_percentage_error(Y_test, pred_test): 
    Y_test, pred_test = np.array(Y_test), np.array(pred_test)
    return np.mean(np.abs((Y_test - pred_test) / Y_test)) * 100

mean_absolute_percentage_error(Y_test, pred_test)

#Linearity
#The Null hypothesis is that the regression is correctly modeled as linear
sm.stats.diagnostic.linear_harvey_collier(lm2)


























