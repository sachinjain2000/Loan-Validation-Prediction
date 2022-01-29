import os
import re
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

import dataMunging
import warnings

warnings.filterwarnings("ignore")

class LogRegWrapper:
    
    def __init__(self):
        self.__subfolder = 'logreg'
        self.__filePrefix = 'logreg'
        self.__model = None

        if not os.path.exists(self.__subfolder):
            os.mkdir(f'./{self.__subfolder}')

    def train(self, data = pd.DataFrame(), y = None):
        
        dump = False
        if data.size == 0:
            dump = True
            COLS_TO_REMOVE = []
            with open(r'./cols_to_remove.txt') as file:
                COLS_TO_REMOVE = file.read().replace(" ","").split(",")
            
            data = pd.read_csv(r'./data/loan-dataset.csv')
            data,y = data.drop(COLS_TO_REMOVE + ['loanid','loanstatus'], axis=1), data.loc[:,'loanstatus'].map({'Y':0,'N':1}).astype('uint8')
        
        X = self.__preProcess(data, exec_type='train')
        
        self.__model = LogisticRegression()
        self.__model.fit(X, y)
        
        if dump:
            pickle.dump(self.__model, open(f'../pickle-models/{self.__filePrefix}-model.pkl', 'wb'))

    def score(self, X, y):
        if not self.__model :
            self.__model = pickle.load(open(f'../pickle-models/{self.__filePrefix}-model.pkl','rb'))
        
        goodX = self.__preProcess(X, 'test',False)

        return self.__model.score(goodX, y)

    def predictProba(self, X):

        data = self.__preProcess(X, 'test', False)
        
        if not self.__model :
            self.__model = pickle.load(open(f'../pickle-models/{self.__filePrefix}-model.pkl','rb'))
        

        return self.__model.predict_proba(data)[:,1]

    def predict(self, X):
        """
            :param:
                X - a dictionary with key as column name and value as column value
        """
        
        data = self.__preProcess(X, 'test')
        
        if not self.__model :
            self.__model = pickle.load(open(f'../pickle-models/{self.__filePrefix}-model.pkl','rb'))

        noProb = self.__model.predict_proba(data)[0,1]

        return 'Y' if noProb <= 0.4 else 'N'
        
    def __preProcess(self, X, exec_type, isDict = True):

        data = pd.DataFrame(X, index=[0]) if (exec_type == 'test' and isDict) else X

        data.columns = list(map(lambda col: re.sub(r"\W|_","",col.lower()),data.columns))

        data = dataMunging.basicMunging(data,imputeCredit_History=False)
        
        data = dataMunging.createDummiesExcept(data=data,filePrefix=f'{self.__subfolder}/{self.__filePrefix}',exec_type=exec_type)
        dataMunging.minMaxScaler(data, ['loanamount', 'loanamountterm'], filePrefix=f'{self.__subfolder}/{self.__filePrefix}',exec_type=exec_type)

        return data

if __name__ == '__main__':
    obj = LogRegWrapper()
    obj.train()
    # print(obj.predict({
    #     'married':'Yes',
    #     'dependents':'1',
    #     "applicantincome":10000,
    #     'coapplicantincome':500,
    #     'loanamount':100,
    #     'loanamountterm':36,
    #     'credithistory':1,
    #     'propertyarea':'Urban'
    # }))