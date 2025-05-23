import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math
import time
import pandas as pd

# Importação de módulos personalizados
from Modules.Helpers import Helper
from Modules.Solvers import Solvers
from Modules.Equations import Equation

class Model:
    def __init__(self, coeffs, bounds, system, labels, datapath, name):
        self.coeffs = coeffs
        self.bounds = bounds
        self.system = system
        self.IND_SIZE = self.count_coeffs()
        self.labels = labels
        self.datapath = datapath
        self.name = name
        self.resolve_data()
        
        
    def resolve_data(self):
        self.df, self.max_data = Helper.load_data(filename=self.datapath, labels=self.labels)
        self.initial_conditions = np.array([self.df[label].iloc[0] for label in self.labels])
        self.t_span = (self.df['t'].iloc[0], self.df['t'].iloc[-1])  # Intervalo de tempo para simulações
        self.t_eval = np.array(self.df['t'])  # Ponto de avaliação dos dados temporais
        self.original = np.array(self.df[self.labels]).T  # Dados originais para cálculo de erro
        
        
    def count_coeffs(self):
        return self.count_coeffs_aux(self.coeffs)
    
    def count_coeffs_aux(self, sub_dict):
        count = 0
        for key, value in sub_dict.items():
            if key == '-':
                continue
            if isinstance(value, dict):
                count += self.count_coeffs_aux(value)
            else:
                count += 1
        return count
    
    def summarize_coeffs(self, coeffs, indent=2, level=0):
        lines = []
        prefix = ' ' * (indent * level)
        for key, value in coeffs.items():
            if key == '-':
                continue
            lines.append(f"{prefix}{key}")
            if isinstance(value, dict):
                lines.append(self.summarize_coeffs(value, indent, level + 1))
        return '\n'.join(lines)
    
        
    def bounds_list(self):
        bounds_list = []
        for key, label in self.coeffs.items():
            bounds_list.append(self.bounds['tau'])
            for key, coeffs in label.items():
                if key != 'tau':
                    bounds_list.append(self.bounds['n'])
                    bounds_list.append(self.bounds['k'])
                    
        return bounds_list
    

    def __repr__(self):
        coeff_summary = self.summarize_coeffs(self.coeffs)
        return (
            f"<Model Summary>\n"
            f"System: {self.system}\n"
            f"Labels: {', '.join(self.labels)}\n"
            f"Data Path: {self.datapath}\n"
            f"Number of Coefficients: {self.IND_SIZE}\n"
            f"Coefficient Structure:\n{coeff_summary}\n"
            f"Bounds: \n{self.bounds}\n"
        )
    
class ModelWrapper:
    
    @staticmethod
    def GRN5():
        labels = ['A', 'B', 'C', 'D', 'E']
        datapath = '../../Data/GRN5_DATA.txt'
        
        coeffs = {
            'A': {
                'E': {'n': None, 'k': None, '-': True},
                'tau': None
            },
            'B': {
                'A': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'C': {
                'B': {'n': None, 'k': None, '-': False},
                'tau': None,
            },
            'D': {
                'C': {'n': None, 'k': None, '-': False},
                'tau': None,
            },
            'E': {
                'D': {'n': None, 'k': None, '-': False},
                'B': {'n': None, 'k': None, '-': False},
                'E': {'n': None, 'k': None, '-': False},
                'tau': None,
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            dA = equation.full_eq(vals, 'A', 'E')
            dB = equation.full_eq(vals, 'B', 'A')
            dC = equation.full_eq(vals, 'C', 'B')
            dD = equation.full_eq(vals, 'D', 'C')
            dE = equation.complex_eqs(vals, 'E', [['+B', '+D'], ['+D', '+E']])

            return [dA, dB, dC, dD, dE]
            
        return Model(coeffs=coeffs, bounds=bounds, system=system, labels=labels, datapath=datapath, name='GRN5')
    
    
    @staticmethod
    def GRN10():
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        datapath = '../../Data/GRN10_DATA.txt'
        
        coeffs = {
            'A': {
                'J': {'n': None, 'k': None, '-': True},
                'tau': None
            },
            'B': {
                'E': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'C': {
                'A': {'n': None, 'k': None},
                'B': {'n': None, 'k': None},
                'F': {'n': None, 'k': None},
                'tau': None
            },
            'D': {
                'F': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'E': {
                'J': {'n': None, 'k': None, '-': True},
                'tau': None
            },
            'F': {
                'A': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'G': {
                'A': {'n': None, 'k': None},
                'B': {'n': None, 'k': None},
                'F': {'n': None, 'k': None},
                'tau': None
            },
            'H': {
                'F': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'I': {
                'G': {'n': None, 'k': None},
                'H': {'n': None, 'k': None},
                'tau': None
            },
            'J': {
                'I': {'n': None, 'k': None, '-': False},
                'tau': None
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            
            dA = equation.full_eq(vals, 'A', 'J')
            dB = equation.full_eq(vals, 'B', 'E')
            dC = equation.complex_eqs(vals, 'C', [
                        ['+B', '-F', '-A'], 
                        ['-B', '+F', '-A'],
                        ['-B', '-F', '+A'],
                        ['+B', '-F', '+A'],
                        ['-B', '+F', '+A'],
                        ['+B', '+F', '+A'],
            ])
            dD = equation.full_eq(vals, 'D', 'F')
            dE = equation.full_eq(vals, 'E', 'J')
            dF = equation.full_eq(vals, 'F', 'A')
            dG = equation.complex_eqs(vals, 'G', [
                        ['+B', '-F', '-A'],
                        ['-B', '+F', '-A'],
                        ['-B', '-F', '+A'],
                        ['+B', '-F', '+A'],
                        ['-B', '+F', '+A'],
                        ['+B', '+F', '+A'] 
            ])
            dH = equation.full_eq(vals, 'H', 'F')
            dI = equation.complex_eqs(vals, 'I', [['+G', '+H']])
            dJ = equation.full_eq(vals, 'J', 'I')

            return [dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ]
            
        return Model(coeffs=coeffs, bounds=bounds, system=system, labels=labels, datapath=datapath, name='GRN10')
    
    
    @staticmethod
    def ABCD():
        labels = ['A', 'B', 'C', 'D']
        datapath = '../../Data/ABCD_DATA.txt'
        
        coeffs = {
            'A': {
                'A': {'n': None, 'k': None},  # nAA & kAA
                'B': {'n': None, 'k': None},  # nAB & kAB
                'D': {'n': None, 'k': None},  # nAD & kAD
                'tau': None                   # tauA
            },
            'B': {
                'C': {'n': None, 'k': None}, # nBC & kBC
                'D': {'n': None, 'k': None}, # nBD & kBD
                'tau': None                  # tauB
            },
            'C': {
                'A': {'n': None, 'k': None}, # nCA & kCA
                'D': {'n': None, 'k': None}, # nCD & kCD
                'tau': None,                 # tauC
            },
            'D': {
                'A': {'n': None, 'k': None}, # nDA & kDA
                'D': {'n': None, 'k': None}, # nDD & kDD
                'tau': None,                 # tauD
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            
            dA = equation.complex_eqs(vals, 'A', [['-A', '-D'], ['+B', '-D'], ['+A', '-B', '+D']])
            dB = equation.complex_eqs(vals, 'B', [['-C'], ['+D']])
            dC = equation.complex_eqs(vals, 'C', [['+D'], ['-A']])
            dD = equation.complex_eqs(vals, 'D', [['-A'], ['-D']])

            return [dA, dB, dC, dD]
            
        return Model(coeffs=coeffs, bounds=bounds, system=system, labels=labels, datapath=datapath, name='ABCD')
    
    @staticmethod
    def ECOLI():
        labels = ['A', 'B', 'C', 'D', 'E']
        datapath = '../../Data/ECOLI_DATA.txt'
        
        coeffs = {
            'A': {
                'A': {'n': None, 'k': None},  # nAA & kAA
                'C': {'n': None, 'k': None},  # nAC & kAC
                'D': {'n': None, 'k': None},  # nAD & kAD
                'E': {'n': None, 'k': None},  # nAE & kAE
                'tau': None                   # tauA
            },
            'B': {
                'A': {'n': None, 'k': None}, # nBA & kBA
                'C': {'n': None, 'k': None}, # nBC & kBC
                'D': {'n': None, 'k': None}, # nBD & kBD
                'E': {'n': None, 'k': None}, # nBE & kBE
                'tau': None                  # tauB
            },
            'C': {
                'D': {'n': None, 'k': None}, # nCD & kCD
                'E': {'n': None, 'k': None}, # nCE & kCE
                'tau': None,                 # tauC
            },
            'D': {
                'C': {'n': None, 'k': None, '-': False}, # nDC & kDC
                'tau': None,                             # tauD
            },
            'E': {
                'A': {'n': None, 'k': None}, # nEA & kEA
                'B': {'n': None, 'k': None}, # nEB & kEB
                'C': {'n': None, 'k': None}, # nEC & kEC
                'E': {'n': None, 'k': None}, # nEE & kEE
                'tau': None,                 # tauD
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            
            dA = equation.complex_eqs(vals, 'A', [['-A', '-D', '-E'], ['-A', '-C', '+E'], ['+A', '+D', '-E']])
            dB = equation.complex_eqs(vals, 'B', [['-A', '-D', '-E'], ['-A', '-C', '+E'], ['+A', '+D', '-E']])
            dC = equation.complex_eqs(vals, 'C', [['+D'], ['+E']])
            dD = equation.full_eq(vals, 'D', 'C')
            dE = equation.complex_eqs(vals, 'E', [['+A', '+B', '+C'], ['+E']])

            return [dA, dB, dC, dD, dE]
            
        return Model(coeffs=coeffs, bounds=bounds, system=system, labels=labels, datapath=datapath, name='ECOLI')
    