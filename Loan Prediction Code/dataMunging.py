import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

import pickle
import utils

def createDummies(data, cols, filePrefix, drop_n_concat = True, exec_type = 'train'):
    if not cols:
        return data
    
    exec_type = exec_type.lower()
    oneHot = None

    if exec_type == 'train':
        oneHot = OneHotEncoder(drop='if_binary')
        oneHot.fit(data[cols])
        pickle.dump(oneHot, open(f'./{filePrefix}_oneHot.pkl','wb'))  #NOTE: writing oneHot as pickle
    
    else:
        oneHot = pickle.load(open(f'./{filePrefix}_oneHot.pkl', 'rb'))

    dummies = pd.DataFrame(
                        oneHot.transform(data.loc[:,cols]).toarray(),
                        columns = oneHot.get_feature_names(cols))
    if drop_n_concat:
        cpy = data.copy()
        cpy.drop(cols, axis = 1, inplace = True)
        cpy.reset_index(inplace=True)
        cpy = pd.concat([cpy, dummies], axis=1)
        return cpy

    return dummies

def selectCategCols(data, include= [], exclude = []):
    """
        :param:
            data - Dataframe to work on
            include - the categorical columns to be included 
            exclude - the categorical columns to ignore
        :return:
            list of categorical columns 
        :raises:
            Assertion Error: if both include and exclude are provided or
                            if none of include or exclude is provided
    """

    assert not (include and exclude), "Either of include or exclude should be provided"
    
    categ_cols = data.select_dtypes(include=["object"]).columns
    
    return [col for col in categ_cols if col in include] if include else [col for col in categ_cols if col not in exclude]

def createDummiesExcept(data, filePrefix, exec_type = 'train', except_ = []):
    """
        :param:
            data - Dataframe to work on
            filePrefix - prefix to be added when writing/reading .pkl file
            exec_type - {'train', 'test'}
            except_ - cols to ignore when applying dummies/one hot encoding
        
        :return:
            dataframe with dummies applied to all the categorical columns in data
            except for those in except_ param
    """
    cpy = data.copy()

    cols_of_interest = selectCategCols(cpy, exclude= except_)
    data_to_feed = createDummies(cpy, cols_of_interest, filePrefix=filePrefix, exec_type= exec_type)
    
    return data_to_feed

def cut(feature, labels, bins = None):
    """
        generates a categorical view of the feature
        The function is smart enough to categorize the feature based on its quantiles
        if bins are not passed
        
        :param:

            feature - Series or 1D array on Which cut operation is to be applied
            labels - list of names to be given to the partitions
            bins - options , used if given else follows quantile parition system
        
        :return:
            categorical view of the feature as 1D-array

        for eg:
            Income = pd.Series(np.random.rand(1000)) 
            cut(Income, ['low', 'moderate','high'])
            it would divide the feature such that 
                Income(min - quantile(0.33))            = 'low'
                Income(quantile(0.33) - quantile(0.66)) = 'moderate'
                Income(quantile(0.66) - max)            = 'high
    """

    assert len(labels) >= 2, "No. of labels should be atleast 2"

    if bins:
        assert len(bins) == len(labels) + 1, "Length mismatch"
    
    feature = np.array(feature)

    if not bins:
        bins = [feature.min(), feature.max()]
        
        n_quantiles = len(labels)-1
        quantile_val = round(1./len(labels), 2)

        for i in range(1, n_quantiles+1):
            bins.insert(i, np.quantile(feature, quantile_val*i))
    
    return np.array(pd.cut(feature, bins, labels=labels))

def ImputeC_H(data, thresh = 0.5):
    
    def fun(x):
        ones = dict(x.value_counts(normalize = True)).get(1.0, 0)
        return 1.0 if ones >= thresh else 0.0
    
    table = pd.pivot_table(data, index = ['TotalIncome_cat'], values=['credithistory'], columns=['propertyarea', 'dependents'], aggfunc=fun)['credithistory']
    data['credithistory'].fillna(data[data['credithistory'].isnull()].apply(
        lambda row:
        table[row['propertyarea']][row['dependents']][row['TotalIncome_cat']],
        axis = 1
    ), inplace = True )

def minMaxScaler(data, cols, filePrefix, exec_type = 'train'):
    """
        scales numeric data

        :param:
            data - Dataframe to be scaled
            cols - list of cols in data that are to be scaled
            filePrefix - prefix to be added when writing/reading .csv file
            exec_type - {'train', 'test'}
        :return:
            None , scales data inplace
    """
    epsilon = 1e+8
    
    if exec_type == 'train':
        dic = {
            'cols':[],
            'min':[],
            'max':[]
        }

        for col in cols:
            dic['cols'].append(col)
            dic['min'].append(data[col].min())
            dic['max'].append(data[col].max())

            denom = data[col].max() - data[col].min()
            data[col] = (data[col] - data[col].min())/(epsilon if denom == 0 else denom)
        
        minMaxInfo = pd.DataFrame(dic)
        minMaxInfo.to_csv(f'./{filePrefix}_min_max.csv', index=False)
    
    else:
        minMaxInfo = pd.read_csv(f'./{filePrefix}_min_max.csv')
        minMaxInfo = minMaxInfo.set_index('cols')

        for col in cols:
            denom = minMaxInfo.loc[col,'max'] - minMaxInfo.loc[col, 'min']
            data.loc[:,col] = (data[col] - minMaxInfo.loc[col,'min'])/(epsilon if denom == 0 else denom)

def basicMunging(data, to_filepath=None, imputeCredit_History = True):
    """
        performs basic data munging i.e.,
        fills na values in Categorical variables with their respective mode values
        and add a new column such that

        TotalIncome_log = log(ApplicantIncome + CoapplicantIncome)

        log is an accepted trick to fix outliners and normalize the data

        :param:
            data - dataFrame containing expected data
            to_filepath - path a csv file where the result is to be written, if specified
            drop_id - specifies whether or not to remove Loan_ID column from the dataset read

        :return:
            returns processed data

        :raises:
            AssertionError : if from_filepath doesn't exist
                            and also if to_filepath's directory doesn't exist
    """

    data['married'].fillna(data['married'].mode()[0], inplace=True)
    data['dependents'].fillna(data['dependents'].mode()[0], inplace=True)
    data['loanamountterm'].fillna(data['loanamountterm'].mode()[0], inplace=True)
    data['loanamount'].fillna(data['loanamount'].median(), inplace = True)

    data['TotalIncome'] = data['applicantincome'] + data['coapplicantincome']
    
    if imputeCredit_History:
        data['TotalIncome_cat'] = cut(data['TotalIncome'], labels=['low', 'moderate', 'high', 'very high'])
        ImputeC_H(data, thresh=0.6)
        data.drop('TotalIncome_cat', axis = 1, inplace = True)
        
    else:
        data['credithistory'].fillna(data['credithistory'].mode()[0], inplace=True)

    
    data['totalincomelog'] = np.log(data['TotalIncome']) #log transform is a trick to fix skwed distributions and outliners
    data.drop(['applicantincome','coapplicantincome','TotalIncome'], axis = 1, inplace = True)
    # uncomment to test the impact of np.log
    """
    # the title includes coefficient of variation = std(x)/mean(x)

    fig, axs = plt.subplots(1,2)
    fig.figsize = (8,8)

    axs[0].set_title(f"Actual distribution {round(data['TotalIncome'].std()/data['TotalIncome'].mean(),2)}")
    axs[0].boxplot(data['TotalIncome'])
    axs[1].set_title(f"log distribution {round(data['TotalIncome_log'].std()/data['TotalIncome_log'].mean(),2)}")
    axs[1].boxplot(data['TotalIncome_log'])
    plt.show()
    """

    data.loc[:,'dependents']= data['dependents'].replace('3+','3').astype('object')
    
    if to_filepath:
        data.to_csv(to_filepath, index = False)
    
    return data
