from src.pipelines.ingestion import ingest
from src.pipelines.transformation import transform
from src.pipelines.feature_engineering import fe
#import os

#import sys
#sys.path.append('./../')

def main():
    ingest("../data/incidentes-viales-c5.csv")
    transform('./output/ingest_df.pkl')
    fe('./output/transformation_df.pkl')