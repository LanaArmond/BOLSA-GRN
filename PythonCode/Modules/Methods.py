import sys
import os
sys.path.append(os.path.abspath("../..")) 

import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math
import time
import pandas as pd

from deap import base, creator, tools, benchmarks
import deap.cma as cma

# Importação de módulos personalizados
from Modules.Helpers import Helper
from Modules.Solvers import Solvers
from Modules.Equations import Equation
from Modules.EvolutionModules import *

from scipy.optimize import differential_evolution


class Method:
    def __init__(self):
        pass

    def run(self, logging, filepath, gens=5000, seed=None, error='SQUARED', solver='RK45', verbose=False, plot_best_ind=False):
        logging.info(f"Starting execution for {self.__class__.__name__}: Solver={solver}, Error={error}, Seed={seed}")
            
        if seed:
            np.random.seed(seed)
        
        start_time = time.time()

        try:
            best_ind = self.execute(logging=logging, solver=solver, seed=seed, error=error, gens=gens, verbose=verbose)

        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
            raise e
            # Se o melhor indivíduo não existir, salvar valores nulos
            
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            if best_ind:
                
                if type(best_ind) is list:
                    best_ind = Individual.list_to_ind(best_ind, model=self.model)
                    
                errors_result = best_ind.calc_all_fitness(solver=solver)  
            else: 
                errors_result = {
                    'ABS': None, 
                    'SQUARED': None, 
                    'MSE': None, 
                    'MABS': None
                }

            result_data = {
                'best_ind': best_ind if best_ind else None,
                'error_type': error,
                'solver': solver,
                'seed': seed,
                'ABS_Fitness': errors_result['ABS'],
                'SQUARED_Fitness': errors_result['SQUARED'],
                'MSE_Fitness': errors_result['MSE'],
                'MABS_Fitness': errors_result['MABS'],
                'execution_time': execution_time,
            }

            logging.info(f"Execution finished for solver: {solver}, seed: {seed}, error: {error} in {execution_time:.2f} seconds.")
            
            if  plot_best_ind:
                Plotter.plot_ind(best_ind, filepath=filepath, name=f"{self.model.name}")
            
            return result_data


    
class CMAES(Method):
    def __init__(self, model, tolerance=1e-4, no_improvement_limit=50, sigma_increase_factor=10, sigma=10, lambda_='auto'):
        super().__init__()
        self.model = model
        self.tolerance = tolerance
        self.sigma = sigma
        self.no_improvement_limit = no_improvement_limit
        self.sigma_increase_factor = sigma_increase_factor
        
        if lambda_ == 'auto':
            self.lambda_ = int(4+(3*np.log(model.IND_SIZE)))
        else:
            self.lambda_ = lambda_
    
    def instantiate_toolbox(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        # Criação do tipo de indivíduo e toolbox
        toolbox = base.Toolbox()

        # Inicializa estratégia CMA-ES
        centroids = Individual.initialize_average_bounds(self.model)
        strategy = cma.Strategy(centroid=centroids, sigma=self.sigma, lambda_=self.lambda_)

        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        # Estatísticas e parâmetros do algoritmo
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda individual: individual.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        return toolbox, hof, strategy, stats
    
    def execute(self, logging, solver, error, seed, gens=5000, verbose=False):
        toolbox, hof, strategy, stats = self.instantiate_toolbox()
        toolbox.register("evaluate", Individual.cma_evaluate, model=self.model, solver=solver, error=error)
        
        population = toolbox.generate()
        Individual.apply_bounds(population, self.model)

        best_fitness = None
        no_improvement_counter = 0
        best_ind = None
        
        for gen in range(gens):
            for i, ind in enumerate(population):
                ind.fitness.values = toolbox.evaluate(ind)

            record = stats.compile(population)
            current_best_fitness = min([ind.fitness.values[0] for ind in population])
            hof.update(population)
            
            if best_fitness is None or current_best_fitness < best_fitness - self.tolerance:
                best_fitness = current_best_fitness
                no_improvement_counter = 0
                best_ind = hof[0]
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= (self.no_improvement_limit + gen / 100):
                strategy.sigma = min(max(strategy.sigma * self.sigma_increase_factor, 1), 15)
                
                if verbose:
                    logging.info(f"Sigma increased: {strategy.sigma}")
                
                
                no_improvement_counter = 0

            toolbox.update(population)
            
            if verbose:
                print(f"Generation {gen}: {record}", end='\r')

            population = toolbox.generate()
            Individual.apply_bounds(population, self.model)
            
        return Individual.list_to_ind(best_ind, self.model)
        
        
class DE(Method):
    def __init__(
        self, 
        model, 
        strategy='best1bin',
        popsize=15,
        mutation=0.8,
        recombination=0.75,
        polish=True,
        disp=True
    ):
        super().__init__()
        self.model = model
        self.strategy = strategy 
        self.popsize = popsize 
        self.mutation = mutation 
        self.recombination = recombination 
        self.polish = polish 
        self.disp = disp 

    def execute(self, logging, solver, error, seed, gens=5000, verbose=False):
        
        def objective_function(params):
            ind = Individual.list_to_ind(params, self.model)
            ind.calculate_fitness()
            return ind.fitness
    
        
        result = differential_evolution(
            objective_function,
            self.model.bounds_list(),
            strategy='best1bin',
            maxiter=gens,
            popsize=15,
            mutation=0.8,
            recombination=0.75,
            seed=seed,
            polish=True,
            disp=True
        )
        
        return Individual.list_to_ind(result.x, self.model)