import pandas as pd
import numpy as np

class Helper:
    def __init__(self):
        pass
    
    
    @staticmethod
    def max_vals(df, labels):
        max_data = {}
        for label in labels:
            max_data[label] =  max(df[label])
            
        return max_data
    
    
    @staticmethod
    def load_data(filename, labels):
        df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['t'] + labels)
        max_data = Helper.max_vals(df, labels)
        return df, max_data