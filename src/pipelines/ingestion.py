import pandas as pd
import numpy as np
import os

def ingest_file(path):
    direction = os.path.join(path, "incidentes-viales-c5" + "." + "csv")
    df = pd.read_csv(direction)
    return df

def drop_cols(df,columnas):
    df.drop(columnas,axis='columns',inplace=True)
    return df

# Crear variable de incidentes_reales
def generate_label(df):
    row_indexes = df[(df['codigo_cierre']=='(N) La unidad de atención a emergencias fue despachada, llegó al lugar de los hechos, pero en el sitio del evento nadie solicitó el apoyo de la unidad')].index    
    row_aux = df[(df['codigo_cierre']=='(F) El operador/a o despachador/a identifican, antes de dar respuesta a la emergencia, que ésta es falsa. O al ser despachada una unidad de atención a emergencias en el lugar de los hechos se percatan que el incidente no corresponde al reportado inicialmente')].index
    row_indexes.union(row_aux)
    df.loc[row_indexes,'incidente_falso']="1"
    df['incidente_falso'] = df['incidente_falso'].replace(np.nan, '0')
    return df

def generate_label_color(df):
    row_indexes = df[(df['codigo_cierre']=='(N) La unidad de atención a emergencias fue despachada, llegó al lugar de los hechos, pero en el sitio del evento nadie solicitó el apoyo de la unidad')].index    
    row_aux = df[(df['codigo_cierre']=='(F) El operador/a o despachador/a identifican, antes de dar respuesta a la emergencia, que ésta es falsa. O al ser despachada una unidad de atención a emergencias en el lugar de los hechos se percatan que el incidente no corresponde al reportado inicialmente')].index
    row_indexes.union(row_aux)
    df.loc[row_indexes,'incidente_falso_color']="red"
    df['incidente_falso_color'] = df['incidente_falso_color'].replace(np.nan, 'blue')
    return df

def save_ingestion(df, path):
    return

