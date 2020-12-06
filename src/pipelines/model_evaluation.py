from src.utils import utils
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score

def load_model(path):
    model = utils.load_df('output/model_loop.pkl')
    return model

def metrics(model):
    #Carga de datos de train y test
    c5_train = utils.load_df('output/fe_df.pkl')
    c5_test = utils.load_df('output/test.pkl')
    #Preparacion de datos test
    c5_test_ = c5_test.copy()
    c5_train_ = c5_train.copy()
    c5_train_ = c5_train_.drop('label',axis=1)
    names=np.array(c5_test_.dia_semana.unique())
    names=np.append(names,np.array(c5_test_.delegacion_inicio.unique()))
    names=np.append(names,np.array(c5_test_.clas_con_f_alarma.unique()))
    names=np.append(names,np.array(c5_test_.tipo_entrada.unique()))
    names=np.append(names,np.array(['año','mes','sin_time', 'cos_time']))   
    #Definicion de transformers
    transformers = [('one_hot', OneHotEncoder(), [ 'dia_semana','delegacion_inicio','clas_con_f_alarma','tipo_entrada']),
        ('año', SimpleImputer(strategy="mean"), ['año_creacion']),
        ('mes', SimpleImputer(strategy="mean"), ['mes']),
        ('impute_sin_time', SimpleImputer(strategy="median"), ['sin_time']),
        ('impute_cos_time', SimpleImputer(strategy="median"), ['cos_time'])]
    col_trans = ColumnTransformer(transformers, remainder="drop", n_jobs=-1, verbose=True)
    col_trans.fit(c5_test_)
    df_input_vars = col_trans.transform(c5_test_)
    X = df_input_vars
    y = c5_test.label.values.reshape(df_input_vars.shape[0],)
    
    X_test = pd.DataFrame.sparse.from_spmatrix(X)
    X_test.columns = names
    X_test
    selection_test = list(c5_train_.columns)
    selection_test
    X_test_ = X_test[selection_test]
    # = model.predict(X_test_)
    #proba = model.predict_proba(X_test_)
    prediction = model[1].best_estimator_.predict(X_test_)
    proba = model[1].best_estimator_.predict_proba(X_test_)
    
    #%matplotlib inline
    #Plot curva ROC
    
    fpr, tpr, thresholds = roc_curve(y, proba[:,1], pos_label=1)
    
    plt.clf()
    plt.plot([0,1],[0,1], 'k--', c="red")
    plt.plot(fpr, tpr)
    plt.title("ROC best RF, AUC: {}".format(roc_auc_score(y, prediction)))
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()
    
    #Plot matriz de confusion

    (confusion_matrix(y, prediction))
    
    #Plot accuracy
    
    accuracy_score(y, prediction)

    # Plot oprecision, recall, thresholds
    
    precision, recall, thresholds_2 = precision_recall_curve(y, proba[:,1], pos_label=1)
    thresholds_2 = np.append(thresholds_2, 1)
    
    #Reporte de metricas
    
    def get_metrics_report(fpr, tpr, thresholds, precision, recall, thresholds_2):
        df_1 = pd.DataFrame({'threshold': thresholds_2,'precision': precision,
                        'recall': recall})
        df_1['f1_score'] = 2 * (df_1.precision * df_1.recall) / (df_1.precision + df_1.recall)
        
        df_2 = pd.DataFrame({'tpr': tpr, 'fpr': fpr, 'threshold': thresholds})
        df_2['tnr'] = 1 - df_2['fpr']
        df_2['fnr'] = 1 - df_2['tpr']
        
        df = df_1.merge(df_2, on="threshold")
        return df
        
    metrics_report = get_metrics_report(fpr, tpr, thresholds, precision, recall, thresholds_2)
    metrics_report
    
    return metrics_report
    #return a
    
    
    #return roc#,precision,recall,metricas

def save_metrics(df,path):
    utils.save_df(df,path)
    
def model_evaluation(path):
    models = load_model(path)
    df = metrics(models)
    save_metrics(df,'output/metricas_offline.pkl')