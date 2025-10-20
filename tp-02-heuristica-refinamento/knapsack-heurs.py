import os
import sys
import numpy as np
import numpy.ma as ma
from scipy import stats
# Você pode precisar instalar a biblioteca 'scikit-posthocs'
# pip install scikit-posthocs
import scikit_posthocs as sp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from autorank import autorank, plot_stats

from mip import Model, xsum, maximize, CBC, OptimizationStatus, BINARY
from glob import glob 
from time import perf_counter as chrono 
NUMERO_MATRICULA = 2025000000
def main():
    # list files in instance folder
    all_insts = glob('./instances/*.kp')

    methods = ["random-swap-1-bit","random2-swap-1-bit","partial-greedy-swap-1-bit","random-swap-1-bit-backtrack"]
    run_time_medians = {m : [] for m in methods}
    obj_val_medians = {m : [] for m in methods}
    # read each instance
    for h, file in enumerate(all_insts):
        # read data file
        assert os.path.isfile(file)==True, f"error finding file {file}"
        inst = CInstance(file) 

        inst_run_times = {m : [] for m in methods} 
        inst_obj_vals = {m : [] for m in methods} 
        # solve each instance 30 times
        for s in range(30):
            # each execution of an instance
            # we use a different seed
            seed = NUMERO_MATRICULA + s * 100
            np.random.seed(seed)
            
            # solve instance by each method
            for m in methods:
                starting_time = chrono()

                sol = CSolution(inst)
                constr = CConstructor()
                ls = CLocalSearch()

                match m:
                    case "random-swap-1-bit":
                            constr.random_solution(sol)
                            ls.swap_one_bit_first_improvement(sol)
                    case "random-swap-2-bits":
                            constr.random_solution(sol)
                            ls.swap_two_bits_first_improvement(sol)
                    case "random2-swap-1-bit":
                            constr.random_solution2(sol)
                            ls.swap_one_bit_first_improvement(sol)
                    case "random2-swap-2-bits":
                            constr.random_solution2(sol)
                            ls.swap_two_bits_first_improvement(sol)
                    case "partial-greedy-swap-1-bit":
                            constr.partial_greedy(sol,.2)
                            ls.swap_one_bit_first_improvement(sol)
                    case "partial-greedy-swap-2-bits":
                            constr.partial_greedy(sol,.2)
                            ls.swap_two_bits_first_improvement(sol)
                    case "random-swap-1-bit-backtrack":
                            constr.random_solution(sol)
                            ls.swap_one_bit_first_improvement_with_backtrack(sol)

                ending_time = chrono()

                inst_run_times[m].append(ending_time - starting_time)
                inst_obj_vals[m].append(sol.obj)
        # for a given instance, get the medians of the run times and obj values  
        # for each method  
        for m in methods:
            run_time_medians[m].append(np.median(inst_run_times[m]))
            obj_val_medians[m].append(np.median(inst_obj_vals[m]))

    # -------------------------------------------
    # 1. Aplicar o Teste de Friedman
    # O teste de Friedman é a primeira etapa para ver se existe alguma diferença
    # estatística entre os métodos.
    alpha = 0.05
    results = np.array([obj_val_medians[m] for m in methods])
    stat_friedman, p_value_friedman = stats.friedmanchisquare(*results)
    print("\n\n Teste de Friedman para os métodos:")
    print(" Hipótese H0: os métodos obtiveram valores de funções objetivos idênticas para os métodos analisados")
    print(f"\testatística de Friedman    : {stat_friedman:.8f}")
    print(f"\tvalor p (p-value)          : {p_value_friedman:.8f}")
    if p_value_friedman < alpha:
        print("\thipótese rejeitada (<0.5): os valores de funções objetivos foram diferentes")

        print("\nExiste uma diferença estatisticamente significativa entre os métodos.")
        print("\nÉ necessário realizar testes PÓS-HOC para identificar quais pares diferem.")

        # -------------------------------------------
        # 2. Aplicar Testes Pós-Hoc (Wilcoxon com Correção de Holm)
        # Holm é geralmente mais poderoso que Bonferroni.
        # O teste abaixo compara os métodos dois a dois pareados:
        post_hoc_results = sp.posthoc_wilcoxon(
            a=results,
            p_adjust='holm' # Aplica a correção de Holm
        )
        
        # Renomeando as colunas e linhas para facilitar a leitura
        post_hoc_results.columns = methods
        post_hoc_results.index = methods
        
        print("\n\n\tResultados pós-Hoc (p-values corrigidos por Holm) ---")
        mat = post_hoc_results.round(4)
        largest_name = max([len(m) for m in methods])

        print("\n\n\tMatriz pós-hoc:")
        print(f"{' ':{largest_name}s} ",end="")
        for i,m1 in enumerate(methods):
            print(f"{m1:^{largest_name}s} ",end="")
        print()
        for i,m1 in enumerate(methods):
            print(f"{m1:{largest_name}s} ",end="")
            for j,m2 in enumerate(methods):
                if i <= j: 
                    print(f"{mat.loc[m1,m2]:{largest_name}.6f}",end="")
                else:
                    print(f"{'':{largest_name}s}",end="")
            print()
        # print("\t",post_hoc_results.round(4))
        # --------------------------
        # verificando quais pares são estatisticamente diferentes
        print("\n\n\tInterpretação da matriz pós-hoc: (sim: significativo/não: não significativo")
        print(f"{' ':{largest_name}s} ",end="")
        for i,m1 in enumerate(methods):
            print(f"{m1:^{largest_name}s} ",end="")  
        print()
        for i,m1 in enumerate(methods):
            print(f"{m1:{largest_name}s} ",end="")
            if i <= j:
                    if mat.loc[m1,m2] < alpha: 
                        print(f"{'sim':^{largest_name}s}",end="")
                    else:
                        print(f"{'não':^{largest_name}s}",end="")
            else:
                    print(f"{'':{largest_name}s}",end="")
            print()

        # -------------------------
        # plotando boxplot 
        df = pd.DataFrame(results.T, columns=methods)
        # convertendo para long format para usar o seaborn
        df_long = df.melt(var_name="método",value_name="função objetivo")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='método', y='função objetivo', data=df_long, hue="método", palette='viridis')
        # adiciona os pontos
        sns.stripplot(data=df_long,\
                     x='método',\
                     y='função objetivo',\
                     hue="método",\
                     palette="dark:.3",\
                     size=3,\
                     alpha=0.5,\
                     legend=False) 
        plt.title('distribuição da função objetivo por método')
        plt.xlabel('método heurístico')
        plt.ylabel('função objetivo (Maior é Melhor)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig('boxplot.pdf')
        plt.show()
        # -------------------------
        # plotando diagrama de clãs: diagrama da diferença crítica
        # calcula o posto (rank) para cada resultado por instância
        # método com o maior valor na instância recebe o posto 1 (o melhor).
        # usamos rankdata e invertemos a ordem (quanto menor o rank, melhor)
        ranking_df = pd.DataFrame(results.T,columns=methods)
        auto_rank_results = autorank(ranking_df,alpha=alpha,order="descending")
        plot_stats(auto_rank_results)
        plt.savefig('cdplot.pdf')
        plt.show()
    else: 
        print("\thipótese aceita (>= 0.5)   : os valores de funções objetivos foram idênticas, sem diferença estatística")
        print("\nNÃO REJEITAMOS H0: Não há diferença estatisticamente significativa entre os métodos")


class CModel():
    def __init__(self,inst):
        self.inst = inst
        N = range(inst.n)
        model = Model('Problema da Mochila',solver_name=CBC)
        # variavel: se o projeto j e incluido na mochila
        x = [model.add_var(var_type=BINARY) for j in N]
        # funcao objetivo: maximizar o retorno
        model.objective = maximize(xsum(inst.p[j] * x[j] for j in N))

        # restricao: a capacidade da mochila deve ser respeitada
        model += xsum(inst.w[j] * x[j] for j in N) <= inst.b
        # desliga a impressao do solver
        model.verbose = 0
        self.x = x
        self.model = model

    def run(self):
        inst = self.inst
        N = range(inst.n)
        model,x = self.model,self.x
        # otimiza o modelo chamando o resolvedor
        status = model.optimize()

        # impressao do resultado
        if status == OptimizationStatus.OPTIMAL:
            print("Optimal solution: {:10.2f}".format(model.objective_value))
            newln = 0
            for j in N:
                if x[j].x > 1e-6:
                    print("{:3d} ".format(j),end='')
                    newln += 1
                    if newln % 10 == 0:
                        newln = 1
                        print()
            print('\n\n')

class CLocalSearch():
    def __init__(self):
        pass

    def swap_bit(self,sol,j):
        inst = sol.inst
        p,w,M,n = inst.p,inst.w,inst.M,inst.n
        b,_b = inst.b,sol._b
        
        oldval,newval = sol.x[j], 0 if sol.x[j] else 1
        delta = p[j] * (newval - oldval)\
              + M * max(0,_b - b)\
              - M * max(0,_b + w[j] * (newval - oldval) - b)
        sol.x[j] = newval 
        sol.obj += delta
        _b += w[j] * (newval - oldval)
        sol._b = _b
        return delta

    def backtrack_parcial(self, sol, k_max):
        inst = sol.inst
        p,w,b,M,n = inst.p,inst.w,inst.b,inst.M,inst.n
        
        S_indices = np.where(sol.x == 1)[0]
        len_S = len(S_indices)
        if len_S == 0:
            return 
        
        k = np.random.randint(1, min(k_max, len_S) + 1) 
        
        items_to_remove = np.random.choice(S_indices, k, replace=False)
        
        for j in items_to_remove:
            sol.x[j] = 0
            sol._b -= w[j]
        sol.get_obj_val() 
        
        S_barra_indices = np.where(sol.x == 0)[0]
        N_indices = S_barra_indices
        
        w_non_zero = np.where(w[N_indices] > 0)[0]
        
        ratio = np.full(len(N_indices), -1.0)
        ratio[w_non_zero] = p[N_indices[w_non_zero]] / w[N_indices[w_non_zero]]
        
        sorted_indices = N_indices[np.argsort(ratio)[::-1]]

        for j in sorted_indices:
            if sol.x[j] == 0 and sol._b + w[j] <= b: 
                sol.x[j] = 1
                sol._b += w[j]
        
        sol.get_obj_val() 
        return sol 

    def swap_one_bit_first_improvement_with_backtrack(self, sol):
        k_max = int(sol.inst.n * 0.2) if sol.inst.n > 0 else 1 
        
        while True:
            sol_before_ls = CSolution(sol.inst)
            sol.copy_solution(sol_before_ls)
            
            self.swap_one_bit_first_improvement(sol) 
            
            if sol.obj > sol_before_ls.obj + 1e-6:
                continue

            sol_after_backtrack = CSolution(sol.inst)
            sol.copy_solution(sol_after_backtrack)
            
            self.backtrack_parcial(sol_after_backtrack, k_max)
            
            if sol_after_backtrack.obj > sol.obj + 1e-6:
                sol_after_backtrack.copy_solution(sol)
            else:
                break
        return

    def swap_one_bit_first_improvement(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        n = inst.n
        N = np.arange(n)

        best_delta = float('inf')

        while best_delta > 0:
             best_delta = -float('inf')
             np.random.shuffle(N)
             improved = False

             for j in N:
                 oldval,newval = sol.x[j], 0 if sol.x[j] else 1
                 delta = p[j] * (newval - oldval)\
                         + M * max(0,_b - b)\
                         - M * max(0,_b + w[j] * (newval - oldval) - b)
                 
                 if delta > 0:
                     sol.x[j] = newval 
                     sol.obj += delta
                     _b += w[j] * (newval - oldval)
                     sol._b = _b
                     improved = True
                     break
             
             if not improved:
                 best_delta = 0
                 
    def swap_two_bits_first_improvement(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        n = inst.n
        N = np.arange(n)

        best_delta = float('inf')

        while best_delta > 0:

             best_delta = -float('inf')
             np.random.shuffle(N)
             improved = False

             h1 = 0
             while h1 < n - 1:
                 j1 = N[h1]
                 oldval1,newval1 = sol.x[j1], 0 if sol.x[j1] else 1

                 h2 = h1 + 1
                 while h2 < n:
                     j2 = N[h2]
                     oldval2,newval2 = sol.x[j2], 0 if sol.x[j2] else 1

                     delta = p[j1] * (newval1 - oldval1)\
                             + p[j2] * (newval2 - oldval2)\
                             + M * max(0,_b - b)\
                             - M * max(0,_b + w[j1] * (newval1 - oldval1) + w[j2] * (newval2 - oldval2) - b)

                     if delta > 0:
                          sol.x[j1] = newval1 
                          sol.x[j2] = newval2 
                          sol.obj += delta
                          _b += w[j1] * (newval1 - oldval1)\
                                + w[j2] * (newval2 - oldval2)
                          sol._b = _b
                          improved = True
                          break
                     h2 += 1
                 
                 if improved:
                      break
                 h1 += 1
             
             if not improved:
                 best_delta = 0

class CConstructor():
    def __init__(self):
        pass

    def random_solution(self,sol):
        inst = sol.inst
        p = np.random.choice(inst.n,1)[0]
        vals = np.random.choice(inst.n,p,replace=False)
        sol.x[:] = 0
        sol.z[:] = -1
        sol.x[vals] = 1
        sol.z[:p] = vals[:]
        sol._b = inst.w[vals].sum()
        sol.get_obj_val()

    def random_solution2(self,sol):
        inst = sol.inst
        N = range(inst.n)
        h = 0
        sol._b = 0
        for j in N:
            val = np.random.choice(2,1)[0]
            sol.x[j] = val
            if val > 0:
                sol._b += inst.w[j]
                sol.z[h] = j
                h += 1
        sol.get_obj_val()

    def partial_greedy(self,sol,alpha):
        inst = sol.inst
        sol.reset()
        N = range(inst.n)

        stop = False
        ls = CLocalSearch()

        rb = np.zeros(inst.n)

        while stop == False:

            for j in N:
                if sol.x[j] == False:
                    delta = ls.swap_bit(sol,j)
                    rb[j] = sol.obj 
                    delta = ls.swap_bit(sol,j)

            masked = ma.masked_array(rb,mask=sol.x)
            maxrb = masked.max()
            minrb = masked.min()
            interval = maxrb - alpha * (maxrb - minrb)

            items = ma.where(masked >= interval)[0]

            if len(items) > 0 and maxrb > 1e-6:
                j = np.random.choice(items,1)[0]
                ls.swap_bit(sol,j)
                if sol.obj < 1e-6:
                    ls.swap_bit(sol,j)
                    stop = True
            else: 
                stop = True

class CSolution():
    def __init__(self,inst):
        self.inst = inst
        self.create_structure()

    def create_structure(self):
        self.x = np.zeros(self.inst.n)
        self.z = np.full(self.inst.n,-1)
        self.obj = 0.0
        self._b = self.inst.b

    def get_obj_val(self):
        inst = self.inst
        p,w,b,M = inst.p,inst.w,inst.b,inst.M
        self.x = self.x.astype(int)
        self._b = (self.x * w).sum()
        self.obj = (self.x * p).sum() - M * max(0,self._b-b)
        return self.obj

    def copy_solution(self,sol):
        sol.x[:] = self.x[:]
        sol.z[:] = self.z[:]
        sol.obj = self.obj
        sol._b = self._b

    def print(self):
        self.get_obj_val()
        print(f'obj  : {self.obj:16.2f}')
        print(f'_b/b : {self._b:16.0f}/{self.inst.b:16.0f}')
        newln = 0
        for j,val in enumerate(self.x):
            if val > 0.9:
                print(f'{j:3d} ',end='')
                newln += 1
                if newln % 10 == 0:
                    newln = 1
                    print()
        print('\n\n')

    def reset(self):
        self.x[:] = 0
        self.z[:] = -1
        self.obj = 0.0
        self._b  = 0.0

class CInstance():
    def __init__(self,filename):
        self.filename = filename
        assert os.path.isfile(filename), 'please, provide a valid file'
        with open(filename,'r') as rf:
            lines = rf.readlines()
            lines = [line for line in lines if line.strip()]
            self.n = int(lines[0])
            self.b = int(lines[1])
            p,w = [],[]
            for h in range(2,self.n+2):
                _p,_w = [int(val) for val in lines[h].split()]
                p.append(_p),w.append(_w)
            self.p,self.w = np.array(p),np.array(w)
        self.M = self.p.sum()

    def print(self):
        print(f'{self.n:9}')
        print(f'{self.b:9}')
        for h in range(self.n):
            print(f'{self.p[h]:4d} {self.w[h]:4d}')

if __name__ == '__main__':
    main()