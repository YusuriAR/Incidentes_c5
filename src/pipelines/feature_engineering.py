import numpy as np
import pandas as pd
from src.utils import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

def load_transformation(path):
    return utils.load_df(path)

def add_column(df):
    df['año_creacion'] = df['fecha_creacion'].dt.year
    return df


def feature_generation(df):
    df.inputs = df.drop(['latitud', 'longitud', 'codigo_cierre','fecha_creacion','incidente_c4'], axis=1)
    return df.inputs

def feature_selection(df):
    seconds_in_day = 24*60*60
    df['sin_time'] = np.sin(2*np.pi*(df.hora_creacion.dt.hour*60*60+df.hora_creacion.dt.minute*60+df.hora_creacion.dt.second)/seconds_in_day)
    df['cos_time'] = np.cos(2*np.pi*(df.hora_creacion.dt.hour*60*60+df.hora_creacion.dt.minute*60+df.hora_creacion.dt.second)/seconds_in_day)
    df = df.drop(['hora_creacion',], axis=1)
    #Ordenar por fecha
    df = df.sort_values(by=['año_creacion','mes'],ascending=True)
    #Dividir la base en train y test
    df_train,df_test = train_test_split(df,test_size=0.2,random_state=1234,shuffle=True)
    #Definimos los nombres  
    names=np.array(df.dia_semana.unique())
    names=np.append(names,np.array(df.delegacion_inicio.unique()))
    names=np.append(names,np.array(df.clas_con_f_alarma.unique()))
    names=np.append(names,np.array(df.tipo_entrada.unique()))
    names=np.append(names,np.array(['año','mes','sin_time', 'cos_time']))   
    #Definicion de transformers
    transformers = [('one_hot', OneHotEncoder(), [ 'dia_semana','delegacion_inicio','clas_con_f_alarma','tipo_entrada']),
                   ('año', SimpleImputer(strategy="mean"), ['año_creacion']),
                   ('mes', SimpleImputer(strategy="mean"), ['mes']),
                   ('impute_sin_time', SimpleImputer(strategy="median"), ['sin_time']),
                   ('impute_cos_time', SimpleImputer(strategy="median"), ['cos_time'])]  
        
    col_trans = ColumnTransformer(transformers, remainder="drop", n_jobs=-1, verbose=True)
    col_trans.fit(df_train)
    df_input_vars = col_trans.transform(df_train)
    X = df_input_vars
    y = df_train.label.values.reshape(df_input_vars.shape[0],)
    # ocuparemos un RF
    classifier = RandomForestClassifier(oob_score=True, random_state=1234)
    # separando en train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # definicion de los hiperparametros que queremos probar
    hyper_param_grid = {'n_estimators': [10,20], 
                        'max_depth': [1, 2, 5],
                        'min_samples_split': [2, 4]}
    #Time Series cross-validator
    tscv = TimeSeriesSplit(n_splits=8)
    gs = GridSearchCV(classifier, 
                               hyper_param_grid, 
                               scoring = 'precision',
                               cv = tscv,
                                
                               n_jobs = -1)
    gs.fit(X_train, y_train) 
    #Importancia con el mejor modelo
    importance = gs.best_estimator_.feature_importances_
    dataset_2 = pd.DataFrame({'importance': importance, 'col_name': names}).sort_values(by='importance',ascending=False)
    dataset_3 = dataset_2[dataset_2['importance']>=0.07]
    columnas_modelo = dataset_3['col_name'].values
    df = pd.DataFrame(X_train.toarray())
    df.columns = names
    df_col = df[columnas_modelo]
    label = pd.DataFrame(y_train,columns=['label'])
    df_modelo = pd.concat([df_col,label],axis=1)        
    return df_modelo,df_test

def save_fe(df):
    #os.chdir('../')
    path = 'output/fe_df.pkl'
    utils.save_df(df, path)

def save_fe_(df):
    #os.chdir('../')
    path = 'output/test.pkl'
    utils.save_df(df, path)

def fe(path):
    df = load_transformation(path)
    df = add_column(df)
    df = feature_generation(df)
    df,df_test = feature_selection(df)
    save_fe(df)
    save_fe_(df_test)