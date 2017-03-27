#coding: utf-8
from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from execute.PreProcess import *

###线性回归####
#读取数据
preProc=PreProcess(r'..\data\lianjia.csv')
data=preProc.process()

X = data[['room_type','size','region','chaoxiang','builtdate']]

y = data['price_union']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

linreg.fit(X, y)

print(linreg.intercept_)
print(linreg.coef_)
print(zip(['room_type','size','region','chaoxiang','builtdate'], linreg.coef_))

#需要预测的参数
predict_value = [11,130.63,14,12,2010]
#预测的房子单价值
predict_outcome = linreg.predict(predict_value)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)

print(linreg.intercept_)
print(linreg.coef_)
print(zip(['room_type','size','region','chaoxiang','builtdate'], linreg.coef_))

y_pred = linreg.predict(X_test)

plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict") 
plt.plot(range(len(y_pred)),y_test,'r',label="test")  
plt.legend(loc="upper right") #显示图中的标签  
plt.xlabel("the number of price_union")  
plt.ylabel('value of price_union')  
plt.show()

#误差评估
from sklearn import metrics

# calculate MAE using scikit-learn
print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
# calculate MSE using scikit-learn
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
# calculate RMSE using scikit-learn
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

# Function to show the resutls of linear fit model
def show_linear_line(X_parameters,Y_parameters):
 # Create linear regression object
 regr = LinearRegression()
 regr.fit(X_parameters, Y_parameters)
 plt.scatter(X_parameters,Y_parameters,color='blue')
 plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
 plt.xticks(())
 plt.yticks(())
 plt.show()

X = data[['room_type']]
X = data[['size']]
X = data[['region']]
X = data[['chaoxiang']]
X = data[['builtdate']]
show_linear_line(X,y)


X = data[['size']]

y = data['price_union']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

linreg.fit(X, y)

print(linreg.intercept_)
print(linreg.coef_)
print(zip(['size'], linreg.coef_))
show_linear_line(X,y)

import seaborn as sns
# visualize the relationship between the features and the response using scatterplots  
sns.pairplot(data, x_vars=['room_type','size','region','chaoxiang','builtdate'], y_vars='price_union', size=7, aspect=0.8)
plt.show()

sns.pairplot(data, x_vars=['room_type','size','region','chaoxiang','builtdate'], y_vars='price_union', size=7, aspect=0.8, kind='reg')
plt.show()