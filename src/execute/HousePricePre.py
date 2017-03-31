#coding: utf-8
from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils

class HousePricePre:
    def __init__(self,data):
        self.data=data

        # Function to show the resutls of linear fit model
    def show_linear_line(self,X_parameters,Y_parameters):
         # Create linear regression object
         regr = self.getModel()
         regr.fit(X_parameters, Y_parameters)
         plt.scatter(X_parameters,Y_parameters,color='blue')
         plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
         plt.xticks(())
         plt.yticks(())
         plt.show()
         
    def getModel(self):
        return LinearRegression()
    
    def getKerasLinearModel(self):
        model = Sequential()
        input_dim_size=self.getXList().__len__()
        model.add(Dense(input_dim=input_dim_size, output_dim=1, init='uniform', activation='linear'))
        model.compile(optimizer='Adam', loss='mse')
        return model

    def getMaxOutLen(self):
        '''
        该函数暂不使用
        '''
        maxBitStr='{0:b}'.format(200000)#输出最大值为20万，房子单价假定不会超过这个数
        outLen=maxBitStr.__len__()
        return outLen

    def getKerasDnnModel(self):
        model = Sequential()

        input_dim_size=self.getXList().__len__()
        model.add(Dense(512, input_shape=(input_dim_size,)))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu')) 
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mse',
              optimizer='Adam')
            #metrics=['accuracy']
        return model
    
    def getXList(self):
        #['hospital','bus','mall','subway','school','officeBuild','room_type','size','region','chaoxiang','builtdate']
        #return ['builtdate','louchentype','loucheng','taxfree','size','hospital','bus','mall','subway','school','officeBuild','region_avg_price']
        return ['builtdate','size','hospital','bus','mall','subway','school','officeBuild','region_avg_price']
        #return ['size']
    
    def gety(self):
        return 'unit_price';
    
    def kerasDnnFit(self,model):
        data=self.data
        X = data[self.getXList()].values#转换为numpy.ndarray
        
        y = data[self.gety()].values;#转换为numpy.ndarray
        input_dim_size=self.getXList().__len__()
        X = X.reshape(X.shape[0], input_dim_size)
        X = X.astype('float32')
        model.fit(X, y,
                    batch_size=128, nb_epoch=3,
                    verbose=1)
        
    def kerasLinearFit(self,model):
        data=self.data
        X = data[self.getXList()].values#转换为numpy.ndarray
        
        y = data[self.gety()].values;#转换为numpy.ndarray
        
        #X = np.linspace(-1, 1, 101)
        #y = 2 * X + np.random.randn(*X.shape) * 0.33 # create a y value which is approximately linear but with some random noise

        model.fit(X, y, nb_epoch=50, verbose=1)
        
        #weights = model.layers[0].get_weights()
        #w = weights[0][0][0]
        #b = weights[1][0]

    def regress(self,model):
        data=self.data
        X = data[self.getXList()]
        
        y = data[self.gety()];
        
        linreg = model
        
        linreg.fit(X, y)
        
        y_pred=linreg.predict(X)
        
        print(linreg.intercept_)
        print(linreg.coef_)
        print(zip(self.getXList(), linreg.coef_))
        
        print('训练的结果如下：')
        #误差评估
        # calculate MAE using scikit-learn
        print("MAE:",metrics.mean_absolute_error(y,y_pred))
        # calculate MSE using scikit-learn
        print("MSE:",metrics.mean_squared_error(y,y_pred))
        # calculate RMSE using scikit-learn
        print("RMSE:",np.sqrt(metrics.mean_squared_error(y,y_pred)))
        
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        linreg.fit(X_train, y_train)
        
        print(linreg.intercept_)
        print(linreg.coef_)
        print(zip(self.getXList(), linreg.coef_))
        
        y_pred2 = linreg.predict(X_test)
        print('测试的结果如下：')
                # calculate MAE using scikit-learn
        print("MAE:",metrics.mean_absolute_error(y_test,y_pred2))
        # calculate MSE using scikit-learn
        print("MSE:",metrics.mean_squared_error(y_test,y_pred2))
        # calculate RMSE using scikit-learn
        print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred2)))
        '''
        plt.figure()
        plt.plot(range(len(y_pred)),y_pred,'b',label="predict") 
        plt.plot(range(len(y_pred)),y_test,'r',label="test")  
        plt.legend(loc="upper right") #显示图中的标签  
        plt.xlabel("the number of price_union")  
        plt.ylabel('value of price_union')  
        plt.show()
        
        import seaborn as sns
        # visualize the relationship between the features and the response using scatterplots  
        sns.pairplot(data, x_vars=['room_type','size','region','chaoxiang','builtdate'], y_vars='price_union', size=7, aspect=0.8)
        plt.show()
        
        sns.pairplot(data, x_vars=['room_type','size','region','chaoxiang','builtdate'], y_vars='price_union', size=7, aspect=0.8, kind='reg')
        plt.show()
        '''