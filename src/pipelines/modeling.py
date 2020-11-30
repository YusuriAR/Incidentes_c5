# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:44:49 2020

@author: DIEGO172
"""

import sys
sys.path.append('./../')
import src
from src import proyecto_1
from src.utils import utils
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

def load_features(path):
     return utils.load_df(path)

def save_models(df):
    #os.chdir('../')
    path = 'output/model_loop.pkl'
    utils.save_df(df, path)
    
def magic_loop(algorithms, df):
    
    y = df['y']
    X = df.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    features = X_train
    labels = y_train
    
    estimators_dict = {'tree': DecisionTreeClassifier(random_state=123),
                  'random_forest': RandomForestClassifier(oob_score=True, random_state=123)}
    
    algorithms_dict = {'tree': 'tree_grid_search',
                  'random_forest': 'rf_grid_search'}
    
    grid_search_dict = {'tree_grid_search': {'max_depth': [1,2,5,None], 
                                         'min_samples_leaf': [2,4]},
                   'rf_grid_search': {'n_estimators': [10,20],  
                                      'max_depth': [1,2,5,None], 
                                      'min_samples_leaf': [2,4]}}
    
    best_estimators = []
    for algorithm in algorithms:
        estimator = estimators_dict[algorithm]
        grid_search_to_look = algorithms_dict[algorithm]
        grid_params = grid_search_dict[grid_search_to_look]
        
        #Time Series cross-validator
        tscv = TimeSeriesSplit(n_splits=8)
        
        gs = GridSearchCV(estimator, grid_params, scoring='precision', cv=tscv, n_jobs=-1)
        
        #train
        gs.fit(features, labels)
        #best estimator
        best_estimators.append(gs)
        
        
    return best_estimators

def modeling(path):
    df = load_features(path)
    algorithms = ['tree','random_forest']
    best = magic_loop(algorithms, df)
    save_models(best)



