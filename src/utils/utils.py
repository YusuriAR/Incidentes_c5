import pickle
import os

def save_df(df,path):
    direction = os.path.join(path, "fe_df" + "." + "pkl")
    pickle.dump(df,open(direction,"wb"))
    
def load_df(path):
    direction = os.path.join(path, "fe_df" + "." + "pkl")
    df = pickle.load(open(direction,"rb"))
    return df