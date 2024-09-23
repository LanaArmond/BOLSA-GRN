import matplotlib.pyplot as plt

class Plotters:
    def __init__():
        pass
    
    # Plotando os resultados
    @staticmethod
    def plot_methods(results, t, methods, labels):
        for method in methods:
            y = results[method]
            plt.figure(figsize=(12, 8))
            
                       
            for i, var in enumerate(labels):
                plt.plot(t, y[i], label=f'{var} - {method}')
            
            plt.xlabel('Tempo')
            plt.ylabel('Concentração')
            plt.title(f'Método: {method}')
            plt.legend()
            plt.show()
            
    
    
    @staticmethod
    def plot_comparison(results, t, df, methods, labels):
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            plt.plot(df['t'], df[label], label=f"{label} -  Dados Originais")

            for method in methods:
                y = results[method]
                plt.plot(t, y[i], label=f'{label} - {method}')

            plt.title('Dados originais vs metodos')
            plt.xlabel('t')
            plt.ylabel('Valores')
            plt.legend()
            plt.show()
