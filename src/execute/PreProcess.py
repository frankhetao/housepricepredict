#coding: utf-8
import pandas as pd
import re

class PreProcess:
    '''
            预 处理需要分析的数据
    '''

    def __init__(self, csvFile):
        self.csvFile=csvFile
        
    def process(self):
        data = pd.read_csv(self.csvFile, error_bad_lines=False,index_col=0)#为空的设置为NaN
        data=data.dropna(how='any')#去掉为NaN的那些行
        data.builtdate=data.builtdate.str.replace('年建', '')#将年份多余的汉字去掉
        return data
        