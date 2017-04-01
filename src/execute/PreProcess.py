#coding: utf-8
import pandas as pd
import re

class PreProcess:
    '''
    预 处理需要分析的数据
    '''

    def __init__(self, csvFile,courtFeature,regionFile,buildDateFile):
        self.csvFile=csvFile
        self.courtFeature=courtFeature
        self.regionFile=regionFile
        self.buildDateFile=buildDateFile
        
    def process(self):
        data = pd.read_csv(self.csvFile, error_bad_lines=False,index_col=0)#为空的设置为NaN
        del data['builtdate']#不用提取的建筑年份，根据小区的建筑年份来
        data=data.dropna(how='any')#去掉为NaN的那些行
        data.loucheng=data.loucheng.str.replace('层', '')#将多余的汉字去掉
        data.taxfree=data.taxfree.str.replace('全税','1')
        data.taxfree=data.taxfree.str.replace('满五','5')
        data.taxfree=data.taxfree.str.replace('满二','2')
        data.louchentype=data.louchentype.str.replace('高区','3')
        data.louchentype=data.louchentype.str.replace('中区','2')
        data.louchentype=data.louchentype.str.replace('低区','1')
        #data=data[data.name=='潍坊四村'] #过滤
        #data.unit_price=data.unit_price/1000#精确到元，MSE太大，因此预测精确到千元即可，例如：一般讨论房子价格，都是单价1.5万元
        data=data[1:]#去掉序号列
        
        dataBuildDate = pd.read_csv(self.buildDateFile, error_bad_lines=False,index_col=0)#为空的设置为NaN
        dataBuildDate.builtdate=dataBuildDate.builtdate.str.replace('年建', '')#将年份多余的汉字去掉
        dataBuildDate=dataBuildDate.dropna(how='any')#去掉为NaN的那些行
        
        dataFeature = pd.read_csv(self.courtFeature, error_bad_lines=False)#为空的设置为NaN
        dataFeature=dataFeature.dropna(how='any')#去掉为NaN的那些行
        dataFeature=pd.merge(dataFeature,dataBuildDate,on=['name'])
        
        data=pd.merge(data,dataFeature,on=['name'])
        
        dataRegion = pd.read_csv(self.regionFile, error_bad_lines=False,index_col=0)#为空的设置为NaN
        data=pd.merge(data,dataRegion,on=['region'])
        
        return data
        