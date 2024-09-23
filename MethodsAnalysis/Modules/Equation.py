import numpy as np

class Equation:
    def __init__(self, coefficients, labels):
        self.coefficients = coefficients
        self.labels = labels
        self.label_to_idx = {label:i for i, label in enumerate(labels)}
    
   
    def main_eq(self, val, nval, kval, minusone):
    
        result = (val**nval / (val**nval + kval**nval))
        
        if minusone:
            return  1 - result
        
        return result

    def full_eq(self, vals, label1, label2, simple_result = False):
        val1 = vals[self.label_to_idx[label1]]
        val2 = vals[self.label_to_idx[label2]]
        nval = self.coefficients[label1][label2]['n']
        kval = self.coefficients[label1][label2]['k']
        tau = self.coefficients[label1]['tau']
        minusone = self.coefficients[label1][label2]['-']
        
        if simple_result:
            return self.main_eq(val2, nval, kval, minusone)
        else:
            return (self.main_eq(val2, nval, kval, minusone) - val1)/tau
    
    def complex_eqs(self, vals, label1, secondary_labels):
        val1 = vals[self.label_to_idx[label1]]
        tau = self.coefficients[label1]['tau']
        
        
        total = 0
        for i, group in enumerate(secondary_labels):        
            subtotal = 0
            for j, sign_label in enumerate(group):
                
                if len(sign_label) == 2:
                    minusone = sign_label[0] == '-'
                    label2 = sign_label[1]
                else:
                    label2 = sign_label
                    minusone = self.coefficients[label1][label2]['-']
                
                
                val2 = vals[self.label_to_idx[label2]]
                nval = self.coefficients[label1][label2]['n']
                kval = self.coefficients[label1][label2]['k']
                
                result = self.main_eq(val2, nval, kval, minusone)
                
                if j == 0:
                    subtotal = result
                else:
                    subtotal *= result
            
            if i == 0:
                total = subtotal
            else:
                total += subtotal
                
        return 1/tau * (total - val1)