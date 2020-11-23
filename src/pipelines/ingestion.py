import pandas as pd
import numpy as np
import os
from src.utils import utils

def ingest_file(path):
    direction = os.path.join(path, "incidentes-viales-c5" + "." + "csv")
    df = pd.read_csv(direction)
    return df

def drop_cols(df):
    columns = ['geopoint','hora_cierre','fecha_cierre','año_cierre','mes_cierre','folio','delegacion_cierre']
    drop_df = df.drop(columns, axis=1)
    return drop_df

def generate_label(df):
    df['label'] = np.where((df['codigo_cierre'].str[:3] == '(F)')|(df['codigo_cierre'].str[:3] == '(N)'), 1, 0)
    return df

def generate_label_incidente(df):
    df['incidente_falso'] = np.where((df['codigo_cierre'].str[:3] == '(F)')|(df['codigo_cierre'].str[:3] == '(N)'), '1', '0')
    return df

def generate_label_incidente_color(df):
    df['incidente_falso_color'] = np.where((df['codigo_cierre'].str[:3] == '(F)')|(df['codigo_cierre'].str[:3] == '(N)'), 'red', 'blue')
    return df

def save_ingestion(df):
    os.chdir('../')
    path = 'output\ingest_df.pkl'
    utils.save_df(df, path)

def ingest(path):
    df = ingest_file(path)
    df = drop_cols(df)
    df = generate_label(df)
    save_ingestion(df)