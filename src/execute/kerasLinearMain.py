from execute.PreProcess import *
from execute.HousePricePre import *

if __name__ == '__main__':
    #Step1 Data
    #读取数据
    preProc=PreProcess(r'..\data\lianjia.csv',r'..\data\CourtFeature.csv',r'..\data\lianjiaregion.csv',r'..\data\CourtBuildDate.csv')
    data=preProc.process()
    
    housePricePre=HousePricePre(data)
    #Step2 Model
    model=housePricePre.getKerasLinearModel()
    
    #Step3 Fit
    housePricePre.kerasLinearFit(model)        
    
    