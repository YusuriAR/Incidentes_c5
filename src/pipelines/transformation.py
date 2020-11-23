import pandas as pd
#import os
from src.utils import utils

def load_ingestion(path):
    return utils.load_df(path)

def date_transformation(col,df):
    df[col]=pd.to_datetime(df[col])
    return df

def numeric_transformation(col,df):
    df[col]=pd.to_numeric(df[col])
    return df

def categoric_trasformation(col,df):
    df = df.astype({col:'category'}) 
    return df

def time_transformation(col,df):
    df[col]=pd.to_datetime(df[col], errors='coerce',format='%H:%M:%S')
    return df

def data_imputation(col,df):
    return df[~df[col].isnull()]

def save_transformation(df):
    #os.chdir('../')
    path = 'output/transformation_df.pkl'
    utils.save_df(df, path)

def transform(path):
    df = load_ingestion(path)
    df = date_transformation('fecha_creacion',df)
    df = time_transformation('hora_creacion',df)
    df = categoric_trasformation('dia_semana',df)
    df = categoric_trasformation('delegacion_inicio',df)
    df = categoric_trasformation('incidente_c4',df)
    df = categoric_trasformation('tipo_entrada',df)
    df = categoric_trasformation('clas_con_f_alarma',df)
    df = categoric_trasformation('mes',df)
    df = data_imputation('delegacion_inicio',df)
    df = data_imputation('latitud',df)
    save_transformation(df)