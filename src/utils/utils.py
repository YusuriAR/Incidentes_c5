import pickle
import os

def save_df(df,path):
    #direction = os.path.join(path, "fe_df" + "." + "pkl")
    pickle.dump(df,open(path,"wb"))
    
def load_df(path):
    #direction = os.path.join(path, "fe_df" + "." + "pkl")
    df = pickle.load(open(path,"rb"))
    return df