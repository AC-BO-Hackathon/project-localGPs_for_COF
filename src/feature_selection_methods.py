import numpy as np     
import csv 
import copy 
import random 
import pandas as pd
import scipy
import sklearn as sk

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error

# User defined files and classes
import utils_dataset as utilsd

# sklearn functions
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso


class feature_selection_algorithms:

    def __init__(self,XX,YY,test_size=0.33,random_state=42):
        
        # Train Data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(XX, YY, test_size=test_size, random_state=random_state)

    # XGBoost
    def xgboost(self, **kwargs):
        
        clf = XGBRegressor(n_estimators=100, learning_rate=0.025, max_depth=20, verbosity=0, booster='gbtree', 
                    reg_alpha=np.exp(-6.788644799030888), reg_lambda=np.exp(-7.450413274554533), 
                    gamma=np.exp(-5.374463422208394), subsample=0.5, objective= 'reg:squarederror', n_jobs=1)  
                           
        paras = clf.get_params()

        clf.fit(self.X_train, self.y_train)        
        return clf
    
    # Features selected by XGBoost
    def selected_features_xgboost(self, descriptors, deep_verbose=False):
        
        clf = self.xgboost()
        score = clf.score(self.X_train, self.y_train)
        if deep_verbose:
            print("XGBoost Training score: ", score)

        scores = cross_val_score(clf, self.X_train, self.y_train,cv=10)
        if deep_verbose:
            print("XGBoost Mean cross-validation score: %.2f" % scores.mean())


        ypred = clf.predict(self.X_test)
        mse = mean_squared_error(self.y_test, ypred)
        if deep_verbose:
            print("XGBoost MSE: %.2f" % mse)
            print("XGBoost RMSE: %.2f" % (mse**(1/2.0)))

        f_importance = clf.get_booster().get_score(importance_type='gain')
        feature_importance_dict={}

        for f,value in f_importance.items():
            feature_index = int(f.split('f')[1])
            feature_importance_dict[descriptors[feature_index]] = value
            if deep_verbose:
                print(f"Column: {feature_index}, descriptor: {descriptors[feature_index]}")
            
        return feature_importance_dict.keys()
                
    
if __name__=="__main__":
    
    print('Feature selection methods are in this class')