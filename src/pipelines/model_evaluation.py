from src.utils import utils
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score
from datetime import datetime
import random

random.seed(123)

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
    prediction = (model[1].best_estimator_.predict_proba(X_test_)[:,1] >= 0.210202).astype(bool)
    proba = model[1].best_estimator_.predict_proba(X_test_)
    
    #%matplotlib inline
    #Plot curva ROC
    
    fpr, tpr, thresholds = roc_curve(y, proba[:,1], pos_label=1)
    
    #plt.clf()
    #plt.plot([0,1],[0,1], 'k--', c="red")
    #plt.plot(fpr, tpr)
    #plt.title("ROC best RF, AUC: {}".format(roc_auc_score(y, prediction)))
    #plt.xlabel("fpr")
    #plt.ylabel("tpr")
    #plt.show()
    
    #Plot matriz de confusion

    (confusion_matrix(y, prediction))
    
    #Plot accuracy
    
    accuracy_score(y, prediction)
    
    #Precision & recall @k
    
    def precision_at_k(y_true, y_scores, k):
        threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
        y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    
        return metrics.precision_score(y_true, y_pred)
    
    def recall_at_k(y_true, y_scores, k):
        threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
        y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    
        return metrics.recall_score(y_true, y_pred)

    def get_top_k(y,y_scores,k):
        array=np.stack((y, y_scores), axis=-1)
        ordena_k = array[np.argsort(array[:, 1])[::-1]]
        k_porc = int(len(y_scores)*k)
        top = ordena_k[:k_porc]
        return top[:,[0]],top[:,[1]]

    def pr_k_curve(y_true, y_scores, save_target):
        k_values = list(np.arange(0.01, 1, 0.01))
        pr_k = pd.DataFrame()
    
        for k in k_values:
            d = dict()
            d['k'] = k
             ## get_top_k es una función que ordena los scores de
             ## mayor a menor y toma los k% primeros
             #top_k_y,top_k_proba = get_top_k(y_true,y_scores, k)
             #d['precision'] = precision_at_k(top_k_y,top_k_proba,k)
             #d['recall'] = recall_at_k(top_k_y,top_k_proba,k)
            d['precision'] = precision_at_k(y_true, y_scores,k)
            d['recall'] = recall_at_k(y_true, y_scores, k)
    
            pr_k = pr_k.append(d, ignore_index=True)
    
        # para la gráfica
        fig, ax1 = plt.subplots()
        ax1.plot(pr_k['k'], pr_k['precision'], label='precision')
        ax1.plot(pr_k['k'], pr_k['recall'], label='recall')
        #ax1.plot([k,k],[1,0], 'k--', c='red')
        
        plt.show()
        
        c5 = utils.load_df('output/ingest_df.pkl')
        min_fecha = '01/01/2014'
        max_fecha = '12/10/2020'
        min_fecha = datetime.strptime(min_fecha, '%d/%m/%Y')
        max_fecha = datetime.strptime(max_fecha, '%d/%m/%Y')
        dias = max_fecha-min_fecha
        dias
        ambulancias = 20
        dias=2476
        acc = c5['dia_semana'].count()
        acc_x_dia = acc/dias
        k = ambulancias / acc_x_dia
        k
        
        plt.axvline(x=k)
        pr_k_curve(y, proba[:,1],0)
        

    #if save_target is not None:
    #    plt.savefig(save_target)
        return pr_k

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
    
    #return roc#,precision,recall,metricas

def save_metrics(df,path):
    utils.save_df(df,path)
    
def model_evaluation(path):
    models = load_model(path)
    df = metrics(models)
    save_metrics(df,'output/metricas_offline.pkl')