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
from Modules.Methods import *

class Execution:
    def __init__(self, model, method, solvers, errors, generations, seeds, output_path, parallel, args):
        self.model = model
        self.method = method
        self.solvers = solvers
        self.errors = errors
        self.generations = generations
        self.seeds = seeds
        self.output_path = output_path
        self.parallel = parallel
        self.args = args
