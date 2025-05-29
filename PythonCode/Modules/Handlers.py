import os
import time
import logging
import multiprocessing as mp
from datetime import datetime
import pandas as pd
import pprint

from Modules.Helpers import Helper  # Você pode usar isso para carregar dados, normalizar, etc.
from Modules.Plotters import Plotter
import traceback

class ExecutionHandler:
    def __init__(self, model, method_class, seeds, solvers, errors, generations, output_path=None, parallel=False, **kwargs):
        self.model = model
        self.method_cls = method_class  # Classe do método, como CMAES
        self.method_name = method_class.__name__.upper()
        self.seeds = seeds
        self.solvers = solvers
        self.errors = errors
        self.generations = generations
        self.output_dir = output_path if output_path else "../../Outputs/"
        self.parallel = parallel
        self.params = kwargs
        
        self.kwargs = kwargs
    
    @property
    def run_path(self):
        return os.path.join(self.output_dir, self.run_id)

    def setup_logger(self):
        os.makedirs(self.run_path, exist_ok=True)
        logger = logging.getLogger(self.run_id)
        logger.setLevel(logging.INFO)

        log_path = os.path.join(self.run_path, "log.txt")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def run_serial(self):
        results = []

        for seed in self.seeds:
            for solver in self.solvers:
                for error in self.errors:
                    self.logger.info(f"Running with seed={seed}, solver={solver}, error={error}")
                    print(f"Running with seed={seed}, solver={solver}, error={error}")
                    result = self.run_single(seed, solver, error)
                    results.append(result)
                    self.save_results(results)

        self.save_results(results)

    def run_parallel(self):
        pool = mp.Pool(mp.cpu_count())
        tasks = [
            (seed, solver, error)
            for seed in self.seeds
            for solver in self.solvers
            for error in self.errors
        ]
        args = [(
            self.model,
            self.method_cls(model=self.model, **self.kwargs),
            self.logger,
            self.run_path,
            self.generations,
            self.method_name,
            self.run_id,
            self.parallel,
            seed,
            solver,
            error,
         ) for seed, solver, error in tasks]
        results = pool.starmap(static_run_single, args)
        
        pool.close()
        pool.join()
        
        self.save_results(results)

    def run_single(self, seed, solver, error):
        try:
            method = self.method_cls(model=self.model, **self.kwargs)
            result = method.run(
                logging=self.logger,
                filepath=self.run_path,
                gens=self.generations,
                seed=seed,
                error=error,
                solver=solver,
                verbose=True
            )
            result.update({
                'model': self.model.name,
                'method': self.method_name,
                'parallel': self.parallel,
                'run_id': self.run_id
            })
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error running seed={seed}, solver={solver}, error={error}: {str(e)}, traceback={traceback.format_exc()}")
            return {
                'model': self.model.name,
                'method': self.method_name,
                'parallel': self.parallel,
                'run_id': self.run_id,
                'seed': seed,
                'solver': solver,
                'error_type': error,
                'ABS_Fitness': None,
                'SQUARED_Fitness': None,
                'MSE_Fitness': None,
                'MABS_Fitness': None,
                'execution_time': None,
            }

    def save_results(self, results):
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.run_path, "results.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to {csv_path}")

    def execute(self):
        
        self.run_id = f"{self.model.name}_{self.method_name}_{'PARALLEL' if self.parallel else 'SERIAL'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = self.setup_logger()
        
        print(f"Starting execution with ID: {self.run_id}")
        
        start = time.time()
        
        if self.parallel:
            self.logger.info(f"Starting execution - Parallel")
            self.run_parallel()
        else:
            self.logger.info(f"Starting execution - Serial")
            self.run_serial()
            
        self.logger.info(f"Execution finished in {time.time() - start:.2f} seconds.")
        
        
        data = pd.read_csv(f"{self.run_path}/results.csv")
        Plotter.plot_comparative_boxplots(data, filepath=self.run_path, filetype='png', name='boxplot_comparison')
    
    
    
    def __repr__(self):
        info = {
            "Model Name": getattr(self.model, "name", str(self.model)),
            "Method Name": self.method_name,
            "Seeds": self.seeds,
            "Solvers": self.solvers,
            "Errors": self.errors,
            "Generations": self.generations,
            "Output Directory": self.output_dir,
            "Parallel Execution": self.parallel,
            "Additional Params": self.params
        }
        return f"<ExecutionHandler Configuration>\n{pprint.pformat(info, indent=2)}"



            
def static_run_single(model, method, logger, run_path, generations, method_name, parallel, run_id, seed, solver, error):
    try:
        result = method.run(
            logging=logger,
            filepath=run_path,
            gens=generations,
            seed=seed,
            error=error,
            solver=solver,
            verbose=True
        )
        result.update({
            'model': model.name,
            'method': method_name,
            'parallel': parallel,
            'run_id': run_id
        })
        
        return result
    
    except Exception as e:
        logger.error(f"Error running seed={seed}, solver={solver}, error={error}: {str(e)}, traceback={traceback.format_exc()}")
        return {
            'model': model.name,
            'method': method_name,
            'parallel': parallel,
            'run_id': run_id,
            'seed': seed,
            'solver': solver,
            'error_type': error,
            'ABS_Fitness': None,
            'SQUARED_Fitness': None,
            'MSE_Fitness': None,
            'MABS_Fitness': None,
            'execution_time': None,
        }