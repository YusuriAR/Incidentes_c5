import pandas as pd

def get_repeated_values(df, col, top):
    top_5 = df.groupby([col])[col]\
                    .count()\
                    .sort_values(ascending = False)\
                    .head(3)
    indexes_top_5 = top_5.index
    
    if ((top == 1) and (len(indexes_top_5) > 0)):
        return indexes_top_5[0]
    elif ((top == 2) and (len(indexes_top_5) > 1)):
        return indexes_top_5[1]
    elif ((top == 3) and (len(indexes_top_5) > 2)):
        return indexes_top_5[2]
    else: 
        return 'undefined'
    
def numeric_profiling(df, col):
    """
    Profiling for numeric columns. 
    
    :param: column to analyze
    :return: dictionary
    """
    profiling = {}

    profiling.update({'max': df[col].max(),
                     'min': df[col].min(),
                     'mean': df[col].mean(),
                     'stdv': df[col].std(),
                     '25%': df[col].quantile(.25),
                     'median': df[col].median(),
                     '75%': df[col].quantile(.75),
                     'kurtosis': df[col].kurt(),
                     'skewness': df[col].skew(),
                     'uniques': df[col].nunique(),
                     'prop_missings': df[col].isna().sum()/df.shape[0]*100,
                     'num_na': df[col].isna().sum(),
                     'top1_repeated': get_repeated_values(df, col, 1),
                     'top2_repeated': get_repeated_values(df, col, 2),
                     'top3_repeated': get_repeated_values(df, col, 3)})
    
    
    return profiling

def category_profiling(df, col):
    """
    Profiling for categoric columns. 
    
    :param: column to analyze
    :return: dictionary
    """
    profiling = {}

    profiling.update({'mode': df[col].mode().values,
                     'num_categories': df[col].nunique(),
                     'categories': df[col].unique(),
                     'uniques': df[col].nunique(),
                     'prop_missings': df[col].isna().sum()/df[col].size*100,
                     'num_na': df[col].isna().sum(),
                     'top1_repeated': get_repeated_values(df, col, 1),
                     'top2_repeated': get_repeated_values(df, col, 2),
                     'top3_repeated': get_repeated_values(df, col, 3)})
    
    return profiling


def datetime_profiling(df, col):
    """
    Profiling for datetime columns. 
    
    :param: column to analyze
    :return: dictionary
    """
    profiling = {}

    profiling.update({'mode': df[col].mode().values,
                     'num_categories': df[col].nunique(),
                     'categories': df[col].unique(),
                     'max': df[col].max(),
                     'min': df[col].min(),
                     'uniques': df[col].nunique(),
                     'prop_missings': df[col].isna().sum()/df[col].size*100,
                     'num_na': df[col].isna().sum(),
                     'top1_repeated': get_repeated_values(df, col, 1),
                     'top2_repeated': get_repeated_values(df, col, 2),
                     'top3_repeated': get_repeated_values(df, col, 3)})
    
    return profiling


# Número de observaciones por categoría y proporción de observaciones por categoría. 

def num_prop(df,column_data):
    # Número de observaciones por categoría
    num=pd.value_counts(df[column_data]) 
    # Proporción de observaciones por categoría
    prop=100 * df[column_data].value_counts() / len(df[column_data])
    # Unión de columnas en una y renombramiento
    columnas_num_prop = round(pd.concat([num, prop], 
                                        keys=['num_obs_cat', 'prop_obs_cat'],
                                        axis=1),2)
    
    return columnas_num_prop

def feature_reduction(df):
    df['codigo_cierre_'] = df['codigo_cierre'].str.slice(0,3)
    return df