import sys
import os
sys.path.append(os.path.abspath("../..")) 

import random
import numpy as np
import copy
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Modules.Plotters import Plotter
from Modules.Helpers import Helper
from Modules.Equations import Equation

# Representa um coeficiente com valor e limites
class Coefficient:
    def __init__(self, bounds):
        self.val = np.random.uniform(*bounds)  # Inicializa com valor aleatório dentro dos limites
        self.bounds = bounds
    
    def __repr__(self):
        return f"val={self.val}"

# Representa um coeficiente usado no CMA-ES com limites.
class CMACoefficient:
    def __init__(self, val, bounds):
        self.bounds = bounds
        self.val = self.limit_val(val)  # Ajusta o valor aos limites

    # Garante que o valor esteja dentro dos limites
    def limit_val(self, val):
        return max(self.bounds[0], min(val, self.bounds[1]))

    def __repr__(self):
        return f"val={self.val}"
    
    
    
# Representa um indivíduo contendo coeficientes e funções para manipulação
class Individual:
    def __init__(self, model):
        self.model = model
        self.fitness = np.inf # Fitness inicializado como infinito
        self.coeffs = copy.deepcopy(self.model.coeffs)
        
        
    # def solve_ivp(self, solver='RK45'):
    #     return integrate.solve_ivp(self.model.system, self.model.t_span, self.model.initial_conditions, method=solver, t_eval=self.model.t_eval, args=(self, self.equation), min_step=0.001).y
    
    def solve_ivp(self, solver='RK45'):
        if solver.upper() == 'ODEINT':
            sol = odeint(
                self.model.system,
                # lambda y, t: self.model.system(t, y, self, self.equation),  # Wrap system for odeint (t first)
                self.model.initial_conditions,
                self.model.t_eval,
                args=(self, self.equation),
                tfirst=True,  # Important: tells odeint the function is (t, y) instead of (y, t)
                hmin=0.001
            )
            
            return sol.T
        else:
            return integrate.solve_ivp(
                self.model.system,
                self.model.t_span,
                self.model.initial_conditions,
                method=solver,
                t_eval=self.model.t_eval,
                args=(self, self.equation),
                min_step=0.001
            ).y
    
    def ind_to_list(self):
        ind_list = []
        for key, label in self.coeffs.items():
            ind_list.append(label['tau'].val)
            for key, coeffs in label.items():
                if key != 'tau':
                    ind_list.append(coeffs['n'].val)
                    ind_list.append(coeffs['k'].val)
        return ind_list
    
        
    def calculate_fitness(self, solver='RK45', error='SQUARED'):
        try:
            y = self.solve_ivp(solver=solver)
            self.fitness = Helper.calculate_error(self.model.original, y, error)
            self.fitness = min(self.fitness, 1e6)
        except:
            # Trata exceções relacionadas ao solver
            print("Overflow")
            self.fitness = 1e6
            
    def calc_all_fitness(self, solver='RK45'):
        y = self.solve_ivp(solver)
        fitness_dict = {}
        for error, error_func in Helper.errors_dict().items():
              fitness_dict[error] = error_func(self.model.original, y,)
       
        return fitness_dict
       
    def initialize_ind(self, solver='RK45', error='SQUARED'):
        for key, label in self.coeffs.items():
            label['tau'] = Coefficient(self.model.bounds['tau'])
            for key, coeffs in label.items():
                if key != 'tau':
                    coeffs['n'] = Coefficient(self.model.bounds['n'])
                    coeffs['k'] = Coefficient(self.model.bounds['k'])
                    
        self.calculate_fitness(solver=solver, error=error)
        
     
    @property
    def equation(self):
        return Equation(self.numerical_coeffs, self.model.labels)
      
    @property
    def numerical_coeffs(self):
        numerical_coeffs = copy.deepcopy(self.coeffs)
        for key, label in numerical_coeffs.items():
            label['tau'] = label['tau'].val
            for key, coeffs in label.items():
                if key != 'tau':
                    coeffs['n'] = int(coeffs['n'].val)
                    coeffs['k'] = coeffs['k'].val
                    
        return numerical_coeffs

            
    @staticmethod        
    def initialize_average_bounds(model):
        array = np.zeros(model.IND_SIZE)
        i = 0
        for key, label in model.coeffs.items():
            array[i] = np.mean(model.bounds['tau'])
            i += 1
            for key, coeffs in label.items():
                if key != 'tau':
                    array[i] = np.mean(model.bounds['n'])
                    array[i+1] = np.mean(model.bounds['k'])
                    i += 2
                    
        return array
        
    
    @staticmethod
    def apply_bounds(population, model):
        for ind in population:
            list_ind = Individual.list_to_ind(ind, model)
            ind[:] = Individual.ind_to_list(list_ind)
    
    @staticmethod    
    def cma_evaluate(list_ind, model, solver='RK45', error='SQUARED'):
        ind = Individual.list_to_ind(list_ind, model)
        ind.calculate_fitness(solver=solver, error=error)
        return ind.fitness,
    
    @staticmethod
    def list_to_ind(list_ind, model):
        i = 0
        ind = Individual(model=model)
        for key, label in ind.coeffs.items():
            label['tau'] = CMACoefficient(list_ind[i], model.bounds['tau'])
            i += 1
            for key, coeffs in label.items():
                if key != 'tau':
                    coeffs['n'] = CMACoefficient(list_ind[i], model.bounds['n'])
                    coeffs['k'] = CMACoefficient(list_ind[i+1], model.bounds['k'])
                    i += 2
        return ind
    
    
    
        
    def __repr__(self):
        coeffs_repr = {k: v for k, v in self.coeffs.items()}
        return f"Individual(fitness={self.fitness}, coeffs={coeffs_repr}, ind_size={self.model.IND_SIZE})"