import pandas as pd

def load_data(file_path):
    print("Loading Dataset....\n")
    data = pd.read_csv(file_path)
    print("Dataset Loaded.\n")
    return data