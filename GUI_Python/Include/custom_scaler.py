
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
from scipy import signal
import pandas as pd
from sklearn.base import TransformerMixin,BaseEstimator
#use class TransformerMixin to apply also the transform when calling .fit (required for the train set)
class custom_MinMaxScaler(TransformerMixin,BaseEstimator): 
    def __init__(self,window):
        print("scaler init done")
        self.window = window
        pass
    def fit(self,X,y=None):
        print("scaler fit done")
        self.scaler =MinMaxScaler(copy=False,feature_range=(-1,1))
        self.scaler.fit(X.reshape(X.shape[0],-1))
        return self

    def transform(self,X,y=None):
        print("scaler transform done\n")
        X_new=self.scaler.transform(X.reshape(X.shape[0],-1))
        X_new = X_new.reshape((X.shape[0],41,45,self.window))
        return X_new

