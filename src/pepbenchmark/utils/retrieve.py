from src.pepbenchmark.metadata import get_dataset_path,DATASET_MAP
import pandas as pd

def retrieve_label_name_list(name):
    path = get_dataset_path(name)
    df = pd.read_csv(path)
    return df.columns.values

