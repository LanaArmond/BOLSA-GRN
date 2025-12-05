import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Dict, Any
from SALib.sample import morris as morris_sampler
from SALib.sample import saltelli
from SALib.analyze import morris, sobol
from Modules.Equations import Equation
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# SHAP
from sklearn.ensemble import RandomForestRegressor
import shap


class SensitivityAnalyzer:
    def __init__(self, model: Any):
        self.model = model
        self.base_coeffs = None
        
    # --- Funções auxiliares ---
    def _flatten_coeffs(self, coeffs: Dict[str, dict]) -> Dict[str, float]:
        # Transforma o dict de coeffs em um dicionário plano
        flat = {}
        for gene, params in coeffs.items():
            for target, subparams in params.items():
                if target == 'tau':
                    flat[f'{gene}.tau'] = subparams
                else:
                    for key, value in subparams.items():
                        if key != '-':  # ignora o sinal da equação
                            flat[f'{gene}->{target}.{key}'] = value
        return flat

    def _rebuild_coeffs(self, base_coeffs: Dict[str, dict], flat_values: Dict[str, float]) -> Dict[str, dict]:
        # Faz reconstrução do dict de coeffs
        coeffs = deepcopy(base_coeffs)
        for key, value in flat_values.items():
            if '->' in key:
                gene, rest = key.split('->')
                target, param = rest.split('.')
                coeffs[gene][target][param] = value
            else:
                gene, param = key.split('.')
                coeffs[gene][param] = value
        return coeffs

    # --- Simulação do modelo ---
    # def _simulate_model(self, coeffs: Dict[str, dict]):
    #     y0 = np.array([self.model.df[label].iloc[0] for label in self.model.labels])
    #     t_eval = np.array(self.model.df['t'])

    #     eq = Equation(coeffs, self.model.labels)

    #     class DummyInd:
    #         pass

    #     dummy = DummyInd()
    #     dummy.model = self.model

    #     def system(t, y):
    #         return self.model.system(t, y, dummy, equation=eq)

    #     sol = solve_ivp(system, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method='LSODA')
    #     return sol.y
    
    def _simulate_model(self, coeffs: Dict[str, dict]):
        y0 = np.array([self.model.df[label].iloc[0] for label in self.model.labels])
        t_eval = np.array(self.model.df['t'])
        eq = Equation(coeffs, self.model.labels)
        
        class DummyInd:
            pass
        dummy = DummyInd()
        dummy.model = self.model
        
        def system(t, y):
            return self.model.system(t, y, dummy, equation=eq)
        
        sol = solve_ivp(system, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method='LSODA')
        
        # Verificar NaN e substituir por 1e6
        if np.isnan(sol.y).any():
            sol.y = np.where(np.isnan(sol.y), 1e6, sol.y)
        
        return sol.y

    # --- Cálculo do erro MSE ---
    def _evaluate_error(self, sim_data: np.ndarray):
        original = self.model.original
        n_sim = sim_data.shape[1]

        original_interp = np.zeros_like(sim_data)
        for i, label in enumerate(self.model.labels):
            f = interp1d(np.linspace(0, 1, original.shape[1]), original[i, :], kind='linear')
            original_interp[i, :] = f(np.linspace(0, 1, n_sim))
            
        error = np.mean((sim_data - original_interp) ** 2)
        
        return 1e6 if np.isnan(error) else error

    # --- Análise Local ---
    def local_sensitivity(self, coeffs: Dict[str, dict], delta: float = 0.05) -> Dict[str, float]:
        # Sensibilidade local via perturbação percentual (delta = variância)
        self.base_coeffs = deepcopy(coeffs)
        flat = self._flatten_coeffs(coeffs)
        sensitivities = {}

        baseline_sim = self._simulate_model(coeffs)
        baseline_err = self._evaluate_error(baseline_sim)
        
        if baseline_err == 0:
            baseline_err = 1e-9

        for key, val in flat.items():
            if val == 0:
                continue
            delta_val = val * delta

            flat_pos = flat.copy()
            flat_pos[key] = val + delta_val
            err_pos = self._evaluate_error(self._simulate_model(self._rebuild_coeffs(coeffs, flat_pos)))

            flat_neg = flat.copy()
            flat_neg[key] = val - delta_val
            err_neg = self._evaluate_error(self._simulate_model(self._rebuild_coeffs(coeffs, flat_neg)))

            derivative = (err_pos - err_neg) / (2 * delta_val)
            sensitivities[key] = abs((val / baseline_err) * derivative)

        sensitivities = dict(sorted(sensitivities.items(), key=lambda x: x[1], reverse=True))
        self.local_results = sensitivities
        return sensitivities

    def plot_local(self, sensitivities: Dict[str, float], top_n: int = 10):
        plt.figure(figsize=(8, 5))
        keys = list(sensitivities.keys())[:top_n]
        values = [sensitivities[k] for k in keys]
        plt.barh(range(len(keys)), values, align='center')
        plt.yticks(range(len(keys)), keys)
        plt.gca().invert_yaxis()
        plt.xlabel('Sensibilidade Normalizada')
        plt.title(f'Top {top_n} Parâmetros - Análise Local')
        plt.tight_layout()
        plt.show()

    # --- Análises de Sensibilidade Global (Morris e Sobol) ---
    def _build_problem(self, coeffs: Dict[str, dict]):
        # Dicionário de entrada do SALib (problem) com base nos coefs
        flat = self._flatten_coeffs(coeffs)
        names = list(flat.keys())
        values = list(flat.values())
        bounds = [(0.5 * v, 1.5 * v) for v in values]  # ±50%
        return {'num_vars': len(names), 'names': names, 'bounds': bounds}

    def morris_sensitivity(self, coeffs: Dict[str, dict] = None, num_trajectories: int = 10):
        coeffs = coeffs or self.base_coeffs
        problem = self._build_problem(coeffs)
        param_values = morris_sampler.sample(problem, N=num_trajectories, num_levels=4)

        Y = []
        print(f"[INFO] Executando análise de Morris com {len(param_values)} amostras...")
        for params in param_values:
            flat = dict(zip(problem['names'], params))
            new_coeffs = self._rebuild_coeffs(coeffs, flat)
            Y.append(self._evaluate_error(self._simulate_model(new_coeffs)))

        result = morris.analyze(problem, param_values, np.array(Y), print_to_console=False)
        result['names'] = problem['names']
        self.morris_results = result
        return result

    def sobol_sensitivity(self, coeffs: Dict[str, dict], N: int = 512, second_order: bool = True):
        flat = self._flatten_coeffs(coeffs)
        problem = {
            'num_vars': len(flat),
            'names': list(flat.keys()),
            'bounds': [[v * 0.5, v * 1.5] for v in flat.values()]
        }

        print(f"[INFO] Gerando {len(flat)} parâmetros e {N} amostras (second_order={second_order})...")
        param_values = saltelli.sample(problem, N=N, calc_second_order=second_order)

        Y = []
        for i, params in enumerate(param_values):
            flat_sample = dict(zip(problem['names'], params))
            new_coeffs = self._rebuild_coeffs(coeffs, flat_sample)
            sim_data = self._simulate_model(new_coeffs)
            Y.append(self._evaluate_error(sim_data))

        Y = np.array(Y)
        print("[INFO] Calculando índices de Sobol...")
        results = sobol.analyze(problem, Y, calc_second_order=second_order, print_to_console=False)
        results['names'] = problem['names']

        # Impressão detalhada
        print("\n=== Resultados da Análise de Sobol ===")
        for i, name in enumerate(problem['names']):
            s1 = results['S1'][i]
            st = results['ST'][i]
            print(f"{name:<25s} | S1={s1:.4f} | ST={st:.4f}")

        if second_order and results.get('S2') is not None:
            print("\n--- Índices de Segunda Ordem (S2) ---")
            names = problem['names']
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    s2_val = results['S2'][i, j]
                    if not np.isnan(s2_val):
                        print(f"{names[i]} x {names[j]} : {s2_val:.4f}")
        else:
            print("\n[INFO] S2 não calculado ou não disponível.")

        self.sobol_results = results
        return results

    def plot_global(self, results: Dict[str, np.ndarray], method: str = "morris", top_n: int = 10):
        if method.lower() == "morris":
            mu_star = results['mu_star']
            names = results['names']
            sorted_idx = np.argsort(mu_star)[::-1]
            names = np.array(names)[sorted_idx][:top_n]
            values = mu_star[sorted_idx][:top_n]
            plt.figure(figsize=(8, 5))
            plt.barh(range(len(names)), values, align='center')
            plt.yticks(range(len(names)), names)
            plt.gca().invert_yaxis()
            plt.xlabel('µ* (Influência Média Absoluta)')
            plt.title("Morris - Sensibilidade Global")
            plt.tight_layout()
            plt.show()

        else:  # Sobol
            names = results['names']
            S1 = results['S1']
            ST = results['ST']
            S2 = results.get('S2', None)

            # --- Plot S1/ST ---
            sorted_idx = np.argsort(S1)[::-1]
            plt.figure(figsize=(10, 5))
            plt.bar(np.array(names)[sorted_idx][:top_n], S1[sorted_idx][:top_n], label="S1")
            plt.bar(np.array(names)[sorted_idx][:top_n], ST[sorted_idx][:top_n], alpha=0.6, label="ST")
            plt.xticks(rotation=90)
            plt.title("Sobol - Índices S1 e ST")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # --- Heatmap S2 ---
            if S2 is not None:
                import seaborn as sns
                plt.figure(figsize=(10, 8))
                sns.heatmap(S2, xticklabels=names, yticklabels=names, cmap="viridis", annot=False)
                plt.title("Sobol - Índices de Segunda Ordem (S2)")
                plt.tight_layout()
                plt.show()
                
    # Análise SHAP
    def shap_sensitivity(self, coeffs, N=2048):
        # 1. gera o problema e as amostras
        problem = self._build_problem(coeffs)
        param_values = saltelli.sample(problem, N=N)
        # return param_values

        # 2. executa simulações
        Y = []
        for params in param_values:
            flat = dict(zip(problem['names'], params))
            new_coeffs = self._rebuild_coeffs(coeffs, flat)
            sim_data = self._simulate_model(new_coeffs)
            Y.append(self._evaluate_error(sim_data))
        Y = np.array(Y)

        # 3. treina surrogate
        surrogate = RandomForestRegressor(n_estimators=400, random_state=0)
        surrogate.fit(param_values, Y)

        # 4. calcula SHAP
        explainer = shap.TreeExplainer(surrogate)
        shap_values = explainer.shap_values(param_values)

        self.shap_explainer = explainer
        self.shap_values = shap_values
        self.shap_X = param_values
        self.shap_names = problem["names"]

        return shap_values

    def plot_shap_summary(self):
        shap.summary_plot(self.shap_values, self.shap_X, feature_names=self.shap_names)

