import numpy as np
import random
from scipy.integrate import solve_ivp
from Modules.Helper import Helper
from Modules.Solvers import Solvers
from Modules.Plotters import Plotters
from Modules.Equation import Equation

class PyGAD_EvolutionStrategy_GRN5:
    def __init__(self, labels, bounds, pop_size=50, generations=100, mutation_rate=0.1, tau=0.5):
        self.labels = labels
        self.bounds = bounds  # Limites MAX e MIN dos coeficientes (n, k, tau)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tau = tau

    # Inicializa uma população com indivíduos aleatórios
    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = {}
            for label in self.labels:
                individual[label] = {
                    'n': np.random.uniform(*self.bounds['n']),
                    'k': np.random.uniform(*self.bounds['k']),
                    'tau': np.random.uniform(*self.bounds['tau'])
                }
            population.append(individual)
        return population

    # Avalia o "fitness" de um indivíduo ao comparar os resultados com os dados reais
    def evaluate_fitness(self, individual, initial_conditions, t_span, t_eval, df, max_data):
        coefficients = self._build_coefficients_dict(individual)
        equation = Equation(coefficients=coefficients, labels=self.labels)

        def system(t, y):
            vals = [Solvers.norm_hardcoded(val, max_data[label]) for val, label in zip(y, self.labels)]
            N_A, N_B, N_C, N_D, N_E = vals

            dA = equation.full_eq(vals, 'A', 'E')
            dB = equation.full_eq(vals, 'B', 'A')
            dC = equation.full_eq(vals, 'C', 'B')
            dD = equation.full_eq(vals, 'D', 'C')
            dE = equation.complex_eqs(vals, 'E', [['+B', '+D'], ['+D', '+E']])
            return [dA, dB, dC, dD, dE]

        result = solve_ivp(system, t_span, initial_conditions, method='RK45', t_eval=t_eval).y
        simulated_data = np.array(result)
        
        mse = np.mean((simulated_data - df[self.labels].values.T) ** 2)
        return -mse

    # Faz a mutação de um indivíduo ao fazer alterações pequenas em seus coeficientes
    def mutate(self, individual):
        new_individual = {}
        for label in self.labels:
            new_individual[label] = {
                'n': np.clip(individual[label]['n'] + np.random.normal(0, self.tau), *self.bounds['n']),
                'k': np.clip(individual[label]['k'] + np.random.normal(0, self.tau), *self.bounds['k']),
                'tau': np.clip(individual[label]['tau'] + np.random.normal(0, self.tau), *self.bounds['tau'])
            }
        return new_individual

    # Seleciona e ranqueia indivíduos baseado na fitness and seleciona os 50% top resultados para reprodução
    def select_parents(self, population, fitnesses):
        sorted_indices = np.argsort(fitnesses)
        selected_parents = [population[i] for i in sorted_indices[-(self.pop_size // 2):]]
        return selected_parents

    # Cria uma nova população ao fazer a reprodução dos pais selecionados
    def reproduce(self, parents):
        new_population = []
        for _ in range(self.pop_size):
            parent1, parent2 = random.sample(parents, 2)
            child = {}
            for label in self.labels:
                if random.random() > 0.5:
                    child[label] = parent1[label]
                else:
                    child[label] = parent2[label]
            new_population.append(child)
        return new_population

    # Loop principal do algoritmo para estratégia evolutiva
    def optimize(self, initial_conditions, t_span, t_eval, df, max_data):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitnesses = [self.evaluate_fitness(ind, initial_conditions, t_span, t_eval, df, max_data) for ind in population]
            print(f'Generation {generation}, Best fitness: {max(fitnesses)}')

            parents = self.select_parents(population, fitnesses)
            population = self.reproduce(parents)
            population = [self.mutate(ind) for ind in population]

        # Retorna o melhor indivíduo
        best_fitness_index = np.argmax(fitnesses)
        best_individual = population[best_fitness_index]
        return best_individual

    # Faz o dicionário de coeficientes a partir de um indivíduo
    def _build_coefficients_dict(self, individual):
        coefficients = {
            'A': {
                'E': {'n': individual['A']['n'], 'k': individual['A']['k'], '-': True},
                'tau': individual['A']['tau']
            },
            'B': {
                'A': {'n': individual['B']['n'], 'k': individual['B']['k'], '-': False},
                'tau': individual['B']['tau']
            },
            'C': {
                'B': {'n': individual['C']['n'], 'k': individual['C']['k'], '-': False},
                'tau': individual['C']['tau']
            },
            'D': {
                'C': {'n': individual['D']['n'], 'k': individual['D']['k'], '-': False},
                'tau': individual['D']['tau']
            },
            'E': {
                'D': {'n': individual['E']['n'], 'k': individual['E']['k'], '-': False},
                'B': {'n': individual['E']['n'], 'k': individual['E']['k'], '-': False},
                'E': {'n': individual['E']['n'], 'k': individual['E']['k'], '-': False},
                'tau': individual['E']['tau']
            }
        }
        return coefficients
