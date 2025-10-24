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
    
# Método DE Adaptativo
class SaDE(Method):
    def __init__(self, model, pop_size=None, max_gens=1000, LP=50):
        super().__init__()
        self.model = model
        self.dim = model.IND_SIZE
        self.pop_size = pop_size if pop_size else 10 * self.dim
        self.max_gens = max_gens
        self.LP = LP

        # Estratégias DE
        self.strategy_names = [
            'rand/1/bin',
            'rand/2/bin',
            'best/1/bin',
            'best/2/bin',
            'current-to-best/1',
            'current-to-rand/1'
        ]
        self.K = len(self.strategy_names)

        # Probabilidades iniciais e médias de parâmetros
        self.p_k = np.ones(self.K) / self.K
        self.CRm = np.full(self.K, 0.5)
        self.Fm = np.full(self.K, 0.5)

        # Contadores de aprendizado
        self.success_count = np.zeros(self.K, dtype=int)
        self.attempt_count = np.zeros(self.K, dtype=int)
        self.success_cr_values = [[] for _ in range(self.K)]
        self.success_f_values = [[] for _ in range(self.K)]

    def _ensure_bounds(self, vec):
        lb, ub = np.array(self.model.bounds_list()).T
        return np.minimum(np.maximum(vec, lb), ub)

    def _sample_CR(self, k):
        cr = np.random.normal(self.CRm[k], 0.1)
        return float(np.clip(cr, 0.0, 1.0))

    def _sample_F(self, k):
        for _ in range(10):
            f = np.random.standard_cauchy() * 0.1 + self.Fm[k]
            if 0 < f <= 1:
                return float(f)
        return float(np.random.uniform(0.1, 0.9))

    def _choose_strategy_index(self):
        return int(np.random.choice(self.K, p=self.p_k))

    def _mutate_and_crossover(self, pop, i, F, CR, strategy_idx, best_idx):
        NP, dim = pop.shape
        idxs = list(range(NP))
        idxs.remove(i)
        r = np.random.choice(idxs, size=5, replace=False)
        x_t = pop[i]
        best = pop[best_idx]

        s = self.strategy_names[strategy_idx]

        if s == 'rand/1/bin':
            a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]
            v = a + F * (b - c)
        elif s == 'rand/2/bin':
            a, b, c, d, e = pop[r[0]], pop[r[1]], pop[r[2]], pop[r[3]], pop[r[4]]
            v = a + F * (b - c + d - e)
        elif s == 'best/1/bin':
            a, b = pop[r[0]], pop[r[1]]
            v = best + F * (a - b)
        elif s == 'best/2/bin':
            a, b, c, d = pop[r[0]], pop[r[1]], pop[r[2]], pop[r[3]]
            v = best + F * (a - b + c - d)
        elif s == 'current-to-best/1':
            a, b = pop[r[0]], pop[r[1]]
            v = x_t + F * (best - x_t) + F * (a - b)
        elif s == 'current-to-rand/1':
            a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]
            v = x_t + np.random.rand() * (a - x_t) + F * (b - c)
        else:
            v = pop[r[0]] + F * (pop[r[1]] - pop[r[2]])

        # Crossover binomial
        jrand = np.random.randint(dim)
        u = np.copy(x_t)
        mask = np.random.rand(dim) < CR
        u[mask] = v[mask]
        u[jrand] = v[jrand]

        return self._ensure_bounds(u)

    def adapt_probabilities(self):
        q = np.where(self.attempt_count > 0,
                     self.success_count / self.attempt_count,
                     0.0)
        q = q + 1e-8
        self.p_k = q / q.sum()
        self.success_count[:] = 0
        self.attempt_count[:] = 0
        for k in range(self.K):
            if self.success_cr_values[k]:
                self.CRm[k] = np.mean(self.success_cr_values[k])
            if self.success_f_values[k]:
                self.Fm[k] = np.mean(self.success_f_values[k])
            self.success_cr_values[k].clear()
            self.success_f_values[k].clear()

    def execute(self, logging, solver, error, seed, gens=None, verbose=False):
        np.random.seed(seed)
        gens = gens if gens else self.max_gens

        bounds = np.array(self.model.bounds_list())
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.pop_size, self.dim))
        fitness = []

        for ind in pop:
            ind_obj = Individual.list_to_ind(ind, self.model)
            ind_obj.calculate_fitness(solver=solver, error=error)
            fitness.append(ind_obj.fitness)
        fitness = np.array(fitness)

        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        best_strategy = None
        best_CR = None
        best_F = None

        for gen in range(1, gens + 1):
            new_pop = pop.copy()
            new_fit = fitness.copy()

            for i in range(self.pop_size):
                # Escolhe estratégia adaptativamente
                k = self._choose_strategy_index()
                CR = self._sample_CR(k)
                F = self._sample_F(k)

                trial = self._mutate_and_crossover(pop, i, F, CR, k, best_idx)
                ind_trial = Individual.list_to_ind(trial, self.model)
                ind_trial.calculate_fitness(solver=solver, error=error)
                tfit = ind_trial.fitness

                # Atualiza contadores de sucesso
                self.attempt_count[k] += 1
                if tfit < fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = tfit
                    self.success_count[k] += 1
                    self.success_cr_values[k].append(CR)
                    self.success_f_values[k].append(F)

                    # Atualiza melhor global
                    if tfit < best_fit:
                        best = trial.copy()
                        best_fit = tfit
                        best_strategy = self.strategy_names[k]
                        best_CR = CR
                        best_F = F

            pop, fitness = new_pop, new_fit
            best_idx = np.argmin(fitness)

            if gen % self.LP == 0:
                self.adapt_probabilities()

            if verbose and gen % max(1, gens // 10) == 0:
                # Evita erro se ainda não houver melhor estratégia
                strategy_str = best_strategy if best_strategy is not None else "-"
                CR_str = f"{best_CR:.3f}" if best_CR is not None else "-"
                F_str = f"{best_F:.3f}" if best_F is not None else "-"

                logging.info(
                    f"[SaDE] Gen {gen}/{gens} | Best Fitness: {best_fit:.6e} "
                    f"| Best Strategy: {strategy_str} | CR={CR_str} | F={F_str}"
                )

        # Log final após todas as gerações
        strategy_str = best_strategy if best_strategy is not None else "-"
        CR_str = f"{best_CR:.3f}" if best_CR is not None else "-"
        F_str = f"{best_F:.3f}" if best_F is not None else "-"

        logging.info(
            f"[SaDE] Final Results:\n"
            f"  -> Best Fitness: {best_fit:.6e}\n"
            f"  -> Best Strategy: {strategy_str}\n"
            f"  -> CR (Crossover Rate): {CR_str}\n"
            f"  -> F (Mutation Factor): {F_str}\n"
            f"  -> Population Size: {self.pop_size}\n"
            f"  -> Strategies Probabilities: {np.round(self.p_k, 3)}\n"
            f"  -> Learning Period (LP): {self.LP}\n"
        )

        return Individual.list_to_ind(best, self.model)


