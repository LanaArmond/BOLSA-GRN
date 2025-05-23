import pandas as pd
import numpy as np

class Helper:
    
    @staticmethod
    def errors_dict():
        return {
            'ABS': Helper.abs_error,
            'MSE': Helper.mse_error,
            'MABS': Helper.mean_abs_error,
            'SQUARED': Helper.squared_error
        }       
    
    
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
    
    @staticmethod
    def abs_error(original, pred):
        return sum(sum(abs(original-pred)))
    
    @staticmethod
    def squared_error(original, pred):
        return sum(sum( (original-pred)**2 ))**(1/2)
    
    @staticmethod
    def mse_error(original, pred):
        return np.mean((original-pred)**2)
    
    @staticmethod
    def mean_abs_error(original, pred):
        return np.mean(abs(original-pred))
    
    @staticmethod
    def calculate_error(original, pred, error):
        return Helper.errors_dict()[error](original, pred)