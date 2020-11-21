import pandas as pd
import os

def ingest_file(path):
    direction = os.path.join(path, "incidentes-viales-c5" + "." + "csv")
    df = pd.read_csv(direction)
    return df
