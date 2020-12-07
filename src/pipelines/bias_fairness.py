from src.utils import utils
import pandas as pd
import numpy as np
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_selected_model(path):
    models = utils.load_df(path)
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
    prediction = (models[1].best_estimator_.predict_proba(X_test_)[:,1] >=0.210635).astype(bool)
    df = pd.DataFrame({'score':prediction, 'label_value': y, 'delegacion': c5_test['delegacion_inicio']})
    df['delegacion'] = df['delegacion'].astype(str)
    return df

    def group(df):
        g = Group()
        xtab, attrbs = g.get_crosstabs(df)
        absolute_metrics = g.list_absolute_metrics(xtab)
        df = [['attribute_name', 'attribute_value']+[col for col in xtab.columns if col in absolute_metrics]].round(4)
        return df
    
    def bias(df):
        g = Group()
        xtab, attrbs = g.get_crosstabs(df)
        bias = Bias()
        bdf = bias.get_disparity_predefined_groups(xtab, original_df=df, 
                                            ref_groups_dict={'delegacion':'GUSTAVO A. MADERO'}, 
                                            alpha=0.05)
        min_bdf = bias.get_disparity_min_metric(xtab, original_df=df)
        df = min_bdf[['attribute_name', 'attribute_value'] +  bias.list_disparities(min_bdf)].round(2)
        return df
    
    def fairness(df):
        g = Group()
        xtab, attrbs = g.get_crosstabs(df)
        bias = Bias()
        bdf = bias.get_disparity_predefined_groups(xtab, original_df=df, 
                                            ref_groups_dict={'delegacion':'GUSTAVO A. MADERO'}, 
                                            alpha=0.05)
        fair = Fairness()
        fdf = fair.get_group_value_fairness(bdf)
        return fdf


def bias_fairness(path):
    df = load_selected_model(path)
    group = group(df)
    bias = bias(df)
    fairness = fairness(df)
    utils.save_df(group_,'output/group.pkl')
    utils.save_df(bias,'output/bias.pkl')
    utils.save_df(fairness,'output/fairness.pkl')
