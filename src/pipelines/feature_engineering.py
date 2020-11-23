import numpy as np
#import os
from src.utils import utils

def load_transformation(path):
    return utils.load_df(path)

def feature_generation(df):
    df.inputs = df.drop(['latitud', 'longitud', 'codigo_cierre','fecha_creacion','incidente_c4'], axis=1)
    return df.inputs

def feature_selection(df):
    seconds_in_day = 24*60*60
    df['sin_time'] = np.sin(2*np.pi*(df.hora_creacion.dt.hour*60*60+df.hora_creacion.dt.minute*60+df.hora_creacion.dt.second)/seconds_in_day)
    df['cos_time'] = np.cos(2*np.pi*(df.hora_creacion.dt.hour*60*60+df.hora_creacion.dt.minute*60+df.hora_creacion.dt.second)/seconds_in_day)
    df = df.drop(['hora_creacion',], axis=1)
    return df

def save_fe(df):
    #os.chdir('../')
    path = 'output/fe_df.pkl'
    utils.save_df(df, path)

def fe(path):
    df = load_transformation(path)
    df = feature_generation(df)
    df = feature_selection(df)
    save_fe(df)
    
    




