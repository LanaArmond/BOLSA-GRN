import matplotlib.pyplot as plt

class Plotter:
    def __init__():
        pass
    
    # Plotando os resultados
    @staticmethod
    def plot_scatter_comparison(results, df, t_eval, t, solvers, labels, filepath=None, name=None, filetype='png'):
        for solver in solvers:
            y = results[solver]
            plt.figure(figsize=(12, 8))
            
                       
            for i, label in enumerate(labels):
                plt.plot(t_eval, y[i], label=f'{label} - {solver}')
                plt.scatter(t, df[label], label=f'Dados reais {label}', marker='o', s=30, alpha=0.7)
            
            plt.xlabel('Tempo')
            plt.ylabel('Concentração')
            plt.title(f'Método: {solver}')
            plt.legend()
            
            if filepath:
                plt.savefig(f"{filepath}/{name}_scatter_comparison.{filetype}")
            else:
                plt.show()
            
    @staticmethod
    def plot_solvers(results, t, solvers, labels):
        for solver in solvers:
            y = results[solver]
            plt.figure(figsize=(12, 8))
            
                       
            for i, var in enumerate(labels):
                plt.plot(t, y[i], label=f'{var} - {solver}')
            
            plt.xlabel('Tempo')
            plt.ylabel('Concentração')
            plt.title(f'Método: {solver}')
            plt.legend()
            plt.show()
            
    
    @staticmethod
    def plot_comparison(results, t, df, solvers, labels, filepath=None, name=None, filetype='png'):
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            plt.plot(df['t'], df[label], label=f"{label} -  Dados Originais")

            for solver in solvers:
                y = results[solver]
                plt.plot(t, y[i], label=f'{label} - {solver}')

            plt.title('Dados originais vs metodos')
            plt.xlabel('t')
            plt.ylabel('Valores')
            plt.legend()
            
            if filepath:
                plt.savefig(f"{filepath}/{name}_{label}_comparison.{filetype}")
            else:
                plt.show()
            

    
    @staticmethod
    def plot_ind(ind, solver='RK45', comparison=False, filepath=None, name=None, filetype='png'):
        model = ind.model
        solvers = [solver]
        results = {}
        results[solver] = ind.solve_ivp(solver=solver)

        Plotter.plot_scatter_comparison(
            results,
             model.df,
             model.t_eval,
             model.df['t'],
             solvers,
             model.labels,
             filepath=filepath,
             name=name,
             filetype=filetype,
        )
                
        # Se comparison=True, plota a comparação usando Plotter
        if comparison:
            Plotter.plot_comparison(
                results=results,
                t=model.t_eval,
                df=model.df,
                solvers=solvers,
                labels=model.labels,
                filepath=filepath,
                name=name,
                filetype=filetype,
            )