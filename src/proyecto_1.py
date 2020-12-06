from src.pipelines.ingestion import ingest
from src.pipelines.transformation import transform
from src.pipelines.feature_engineering import feature_engineering
from src.pipelines.modeling import modeling
from src.pipelines.model_evaluation import model_evaluation
#from src.pipelines.bias_fairness import bias_fairness

#import os

#import sys
#sys.path.append('./../')

def main():
    ingest("../data/incidentes-viales-c5.csv")
    transform('./output/ingest_df.pkl')
    feature_engineering('./output/transformation_df.pkl')
    modeling('./output/fe_df.pkl')
    model_evaluation('./output/model_loop.pkl')
    #bias_fairness('./output/model_loop.pkl')