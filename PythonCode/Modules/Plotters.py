import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    def __init__():
        pass
    
    
    @staticmethod
    def plot_comparative_boxplots(data, filepath=None, filetype='png', name='boxplot_comparison'):
        sns.set_theme(style="whitegrid")
        metrics = ['ABS_Fitness', 'SQUARED_Fitness', 'MSE_Fitness', 'MABS_Fitness']

        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=data,
                x='method',
                y=metric,
                hue='error_type',
                showfliers=True
            )
            plt.title(f'Comparação de {metric.capitalize()} por Método e Tipo de Erro')
            plt.xlabel('Método')
            plt.ylabel(metric.capitalize())
            plt.legend(title='Tipo de Erro')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if filepath:
                plt.savefig(f"{filepath}/{name}_{metric}_comparacao.{filetype}")
                plt.close()
            else:
                plt.show()

        # Tempo de execução
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=data,
            x='method',
            y='execution_time',
            hue='error_type',
            showfliers=True,
            palette='Set2'
        )
        plt.title('Comparação de Tempo de Execução por Método e Tipo de Erro')
        plt.xlabel('Método')
        plt.ylabel('Tempo de Execução')
        plt.legend(title='Tipo de Erro')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if filepath:
            plt.savefig(f"{filepath}/{name}_execution_time_comparacao.{filetype}")
            plt.close()
        else:
            plt.show()
    
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