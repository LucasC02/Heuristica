import os
import sys
import numpy as np
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY
from time import perf_counter as pc
import math

np.random.seed(5000)

# Parâmetros globais para o Simulated Annealing
INITIAL_TEMPERATURE = 100.0 # T_{0}
COOLING_RATE = 0.95 # kappa

def main():
    assert len(sys.argv) > 1, 'please,provide a data file'
    inst = CInstance(sys.argv[1])
    #inst.print()

    # Usando o método construtivo 'greedy' como o melhor do trabalho anterior (PS do PDF)
    sol_greedy = CSolution(inst)
    constr = CConstructor()
    print('construction phase: greedy ')
    constr.greedy(sol_greedy)
    sol_greedy.print()
    
    ls = CBuscaLocal(INITIAL_TEMPERATURE, COOLING_RATE)
    
    MAX_TIME = 1000 # Tempo máximo para a busca (apenas para referência, o SILS usa iterações)
    H_MAX = 50 # Número máximo de iterações do SILS
    K_MAX = 10 # Número máximo de perturbações/agitações na mesma vizinhança
    
    # 1. Teste de Busca Local (VND vs RVND) com Critério Guloso
    print('\n--- SILS (Smart ILS) com Critério Guloso ---')
    
    # SILS com VND
    sol_sils_vnd_greedy = CSolution(inst)
    sol_sils_vnd_greedy.copy_solution(sol_greedy)
    print('SILS com VND (Critério: Guloso)')   
    ls.ils(sol_sils_vnd_greedy, max_time=MAX_TIME, max_iterations=H_MAX, max_perturbation=K_MAX, local_search='vnd', acceptance_criterion='greedy')
    sol_sils_vnd_greedy.print()

    # SILS com RVND
    sol_sils_rvnd_greedy = CSolution(inst)
    sol_sils_rvnd_greedy.copy_solution(sol_greedy)
    print('SILS com RVND (Critério: Guloso)')   
    ls.ils(sol_sils_rvnd_greedy, max_time=MAX_TIME, max_iterations=H_MAX, max_perturbation=K_MAX, local_search='rvnd', acceptance_criterion='greedy')
    sol_sils_rvnd_greedy.print()

    # 2. Teste de Critério de Aceitação (Gulosa vs SA)
    print('\n--- SILS (Smart ILS) com VND e Critério SA ---')
    
    # SILS com VND e Critério SA
    sol_sils_vnd_sa = CSolution(inst)
    sol_sils_vnd_sa.copy_solution(sol_greedy)
    print('SILS com VND (Critério: Simulated Annealing)')   
    ls.ils(sol_sils_vnd_sa, max_time=MAX_TIME, max_iterations=H_MAX, max_perturbation=K_MAX, local_search='vnd', acceptance_criterion='sa')
    sol_sils_vnd_sa.print()

    # SILS com RVND e Critério SA
    sol_sils_rvnd_sa = CSolution(inst)
    sol_sils_rvnd_sa.copy_solution(sol_greedy)
    print('SILS com RVND (Critério: Simulated Annealing)')   
    ls.ils(sol_sils_rvnd_sa, max_time=MAX_TIME, max_iterations=H_MAX, max_perturbation=K_MAX, local_search='rvnd', acceptance_criterion='sa')
    sol_sils_rvnd_sa.print()
    
    mod = CModel(inst)
    mod.run()
     

class CModel():
    def __init__(self,inst):
        self.inst = inst
        inst = self.inst
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

class CBuscaLocal():
    def __init__(self, initial_temp, cooling_rate):
        self.T_0 = initial_temp
        self.kappa = cooling_rate

    def acceptance_criterion_sa(self, S, S_prime_prime, T):
        # Algoritmo 4.0.3
        
        # O ILS busca a maximização (f(S) > f(S*)). 
        # Simulated Annealing busca minimização. Vamos inverter o sinal
        # do delta para maximização: Delta = - (f(S'') - f(S)). 
        # Assim, se f(S'') > f(S), Delta é positivo.
        
        # Para o problema de maximização (f(S) > f(S*)):
        # f(S'') - f(S) -> O quanto melhor S'' é que S. Se > 0, S'' é melhor.
        Delta = S_prime_prime.obj - S.obj # 2. Delta <- f(S'') - f(S)
        
        # 3. delta <- Rand(0,1) / // valor aleatório entre 0 e 1 (no PDF: theta e 1)
        delta = np.random.uniform(0, 1)

        if Delta > 0:
            # Se for melhor, aceita
            S.copy_solution(S_prime_prime) # 5. S <- S''
            return True
        elif T > 0 and delta < math.exp(Delta / T): # 4. if delta < e^(Delta / T) then
            # Se for pior (Delta <= 0), aceita com probabilidade e^(Delta / T) (se for maximização)
            # Como Delta <= 0, e^(Delta/T) <= 1.
            # No SA clássico (minimização) é -Delta. Para maximização, é só Delta.
            S.copy_solution(S_prime_prime) # 5. S <- S''
            return True
            
        return False # Não aceita

    def acceptance_criterion_greedy(self, S, S_prime_prime):
        # Critério Guloso: aceita S'' apenas se for melhor que S.
        if S_prime_prime.obj > S.obj:
            S.copy_solution(S_prime_prime)
            return True
        return False
        
    def rvnd(self,sol, strategy='best'):
        # Implementação RVND simplificada: Melhoria no RVND original do arquivo
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        
        N_list = ['1b', '2b'] if strategy == 'best' else ['1', '2']
        n_max = len(N_list)
        
        k = 1
        while (k <= n_max):
            if k == 1:
                randn = np.arange(n_max)
                np.random.shuffle(randn)

            viz_key = N_list[randn[k-1]]
            
            # Mapeamento das funções
            if viz_key == '1':
                self.swap_one_bit(sol)
            elif viz_key == '2':
                self.swap_two_bits(sol)
            elif viz_key == '1b':
                self.swap_one_bit_best_improvement(sol)
            elif viz_key == '2b':
                self.swap_two_bits_best_improvement(sol)

            if sol.obj > solstar.obj:
               solstar.copy_solution(sol)
               k = 1
            else:
               k += 1

        sol.copy_solution(solstar)

    def vnd(self,sol,strategy='best'):
        # Implementação VND simplificada:
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)

        h = 1
        while (h <= 2):
            if strategy == 'best':
               if h == 1:
                   self.swap_one_bit_best_improvement(sol)
               elif h == 2:
                   self.swap_two_bits_best_improvement(sol)
               else:
                    break
            else: # First Improvement
               if h == 1:
                  self.swap_one_bit(sol)
               elif h == 2:
                  self.swap_two_bits(sol)
               else:
                  break

            if sol.obj > solstar.obj:
               solstar.copy_solution(sol)
               h = 1
            else:
               h += 1
        
        sol.copy_solution(solstar) # Algoritmo 3.0.1 retorna o ótimo local S_star

    
    def vns(self,sol,max_time,strategy='best'):
        # Mantido como está no código original, usado para referência
        crono = Crono()
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        while crono.get_time() < max_time:
            h = 1
            while h <= 2:
                if h == 1:
                    self.random_swap_one_bit(sol)
                elif h == 2:
                    self.random_swap_two_bits(sol)
                else:
                    break
                self.vnd(sol,strategy=strategy)

                if sol.obj > solstar.obj:
                   solstar.copy_solution(sol)
                   h = 1
                else:
                   h += 1
        sol.copy_solution(solstar)

    def ils(self,sol,max_time,max_iterations,max_perturbation = 5, local_search='vnd', acceptance_criterion='greedy'):
        # Smart Iterated Local Search (SILS) - Algoritmo 4.0.2
        crono = Crono()
        solstar = CSolution(sol.inst)
        
        # Configuração da Busca Local
        ls_func = self.vnd if local_search == 'vnd' else self.rvnd
        
        # 1. S <- BuscaLocal(S) // Encontra um ótimo local inicial
        ls_func(sol) 
        
        # 2. S* <- S
        solstar.copy_solution(sol)

        # 3. h <- 0
        h = 0
        
        # 4. p <- 1 // nível de perturbação
        pert_level = 1
        
        # Variável de Temperatura (para SA)
        T = self.T_0
        
        # 5. while h < h_max do
        while h < max_iterations and crono.get_time() < max_time:
          
            # 6. k <- 0
            k = 0
            
            # 7. while k < k_max do
            while k < max_perturbation:
                 # Debug/Print
                 print(f'h {h:3d} k {k:3d} p {pert_level:3d} S* {solstar.obj:12.2f} S {sol.obj:12.2f} T {T:10.2f} {crono.get_time():10.2f}s')
                 
                 # 8. S' <- Perturbação(S, p)
                 sol_prime = CSolution(sol.inst)
                 sol_prime.copy_solution(sol) # Começa do ótimo local S
                 self.perturbation(sol_prime, pert_level)
                 
                 # 9. S'' <- BuscaLocal(S') // VND/RVND na solução perturbada
                 sol_prime_prime = CSolution(sol.inst)
                 sol_prime_prime.copy_solution(sol_prime)
                 ls_func(sol_prime_prime) 
                 
                 # 10. S <- CritérioAceitação(S, S'')
                 accepted = False
                 if acceptance_criterion == 'sa':
                    # O SILS não usa a temperatura no Critério de Aceitação 
                    # para aceitar S'' como novo S, mas sim T.
                    # Vamos passar a temperatura atual T.
                    accepted = self.acceptance_criterion_sa(sol, sol_prime_prime, T)
                 else: # 'greedy'
                    accepted = self.acceptance_criterion_greedy(sol, sol_prime_prime)
                 
                 # 11. if f(S) > f(S*) then
                 if sol.obj > solstar.obj:
                    # 12. S* <- S // atualiza a melhor solução global
                    solstar.copy_solution(sol)
                    # 13. h <- 1
                    h = 1
                    # 14. k <- 0
                    k = 0
                    # 15. p <- 1
                    pert_level = 1
                 # 16. else
                 else:
                    # 17. h <- h + 1
                    h += 1
                 
                 # 19. k <- k + 1
                 k += 1
                 
            # 21. p <- p + 1
            pert_level += 1

            # Atualiza a temperatura se SA for usado (Algoritmo 4.0.3, passo 7)
            if acceptance_criterion == 'sa':
                T = self.kappa * T
                # Para evitar T=0 e e^(Delta/T) ser indefinido
                if T < 1e-6: T = self.T_0 # Reinicia a temperatura ou para.

        sol.copy_solution(solstar)

    def perturbation(self,sol,pert_level):
        # Perturbação de baixa intensidade: 'p' chamadas a random_swap_one_bit
        for _ in range(pert_level):
            self.random_swap_one_bit(sol)

    # ... (Restante das funções de busca local e auxilares, mantidas do código original)

    def random_swap_one_bit(self,sol):
        inst = sol.inst
        n,p,w,b,_b,M = inst.n,inst.p,inst.w,inst.b,sol._b,inst.M
        idx = np.random.randint(n)
        oldval,newval = sol.x[idx], 0 if sol.x[idx] else 1
        delta = p[idx] * (newval - oldval)\
              + M * max(0,_b - b)\
              - M * max(0,_b + w[idx] * (newval - oldval) - b)
        sol.x[idx] = newval 
        sol.obj += delta
        _b += w[idx] * (newval - oldval)
        sol._b = _b
        
    def random_swap_two_bits(self,sol):
        inst = sol.inst
        n,p,w,b,_b,M = inst.n,inst.p,inst.w,inst.b,sol._b,inst.M
        idx1,idx2 = np.random.choice(n,size=2,replace=False)
        oldval1,newval1 = sol.x[idx1], 0 if sol.x[idx1] else 1
        oldval2,newval2 = sol.x[idx2], 0 if sol.x[idx2] else 1
        delta = p[idx1] * (newval1 - oldval1)\
              + p[idx2] * (newval2 - oldval2)\
              + M * max(0,_b - b)\
              - M * max(0,_b + w[idx1] * (newval1 - oldval1) \
              + w[idx2] * (newval2 - oldval2) - b)
        sol.x[idx1] = newval1 
        sol.x[idx2] = newval2 
        sol.obj += delta
        _b += w[idx1] * (newval1 - oldval1)\
            + w[idx2] * (newval2 - oldval2)
        sol._b = _b
    
    def swap_one_bit(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        N = np.arange(inst.n)

        np.random.shuffle(N)
        delta = float('inf')
        while delta > 0:
              delta = -float('inf')

              for j in N:
                  oldval,newval = sol.x[j], 0 if sol.x[j] else 1
                  delta = p[j] * (newval - oldval)\
                        + M * max(0,_b - b)\
                        - M * max(0,_b + w[j] * (newval - oldval) - b)
                  if delta > 0:
                     oldval,newval = sol.x[j], 0 if sol.x[j] else 1
                     sol.x[j] = newval 
                     sol.obj += delta
                     _b += w[j] * (newval - oldval)
                     sol._b = _b
                     break
   
    def swap_two_bits(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        n = inst.n
        N = np.arange(n)
        np.random.shuffle(N)

        delta = float('inf')

        while delta > 0:

              delta = -float('inf')

              h1 = 0
              found_improvement = False
              while h1 < n - 1:
                  j1 = N[h1]

                  h2 = h1 + 1
                  while h2 < n:
                      j2 = N[h2]
                      oldval1,newval1 = sol.x[j1], 0 if sol.x[j1] else 1
                      oldval2,newval2 = sol.x[j2], 0 if sol.x[j2] else 1

                      delta = p[j1] * (newval1 - oldval1)\
                            + p[j2] * (newval2 - oldval2)\
                            + M * max(0,_b - b)\
                            - M * max(0,_b + w[j1] * (newval1 - oldval1) + w[j2] * (newval2 - oldval2) - b)

                      if delta > 0:
                         oldval1,newval1 = sol.x[j1], 0 if sol.x[j1] else 1
                         oldval2,newval2 = sol.x[j2], 0 if sol.x[j2] else 1
                         sol.x[j1] = newval1 
                         sol.x[j2] = newval2 
                         sol.obj += delta
                         _b += w[j1] * (newval1 - oldval1)\
                             + w[j2] * (newval2 - oldval2)
                         sol._b = _b
                         found_improvement = True
                         break
                      h2 += 1
                  if found_improvement:
                      break
                  h1 += 1
              if not found_improvement:
                  delta = -1 

    def swap_one_bit_best_improvement(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        N = np.arange(inst.n)

        best_delta = float('inf')
        best_j = -1

        while best_delta > 0:
              best_delta = -float('inf')

              for j in N:
                  oldval,newval = sol.x[j], 0 if sol.x[j] else 1
                  delta = p[j] * (newval - oldval)\
                        + M * max(0,_b - b)\
                        - M * max(0,_b + w[j] * (newval - oldval) - b)
                  if delta > best_delta:
                      best_delta = delta
                      best_j = j

              if best_delta > 0:
                  oldval,newval = sol.x[best_j], 0 if sol.x[best_j] else 1
                  sol.x[best_j] = newval 
                  sol.obj += best_delta
                  _b += w[best_j] * (newval - oldval)
                  sol._b = _b

    def swap_two_bits_best_improvement(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        n = inst.n
        N = np.arange(n)

        best_delta = float('inf')
        best_j1 = -1
        best_j2 = -1

        while best_delta > 0:

              best_delta = -float('inf')

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

                      if delta > best_delta:
                         best_delta = delta
                         best_j1 = j1
                         best_j2 = j2

                      h2 += 1
                  h1 += 1

              if best_delta > 0:
                  oldval1,newval1 = sol.x[best_j1], 0 if sol.x[best_j1] else 1
                  oldval2,newval2 = sol.x[best_j2], 0 if sol.x[best_j2] else 1
                  sol.x[best_j1] = newval1 
                  sol.x[best_j2] = newval2 
                  sol.obj += best_delta
                  _b += w[best_j1] * (newval1 - oldval1)\
                      + w[best_j2] * (newval2 - oldval2)
                  sol._b = _b

class CConstructor():
    def __init__(self):
        pass

    def random_solution2(self,sol):
        inst = sol.inst
        p = np.random.choice(inst.n,1)[0]
        vals = np.random.choice(inst.n,p,replace=False)
        sol.x[:] = 0
        sol.z[:] = -1
        sol.x[vals] = 1
        sol.z[:p] = vals[:]
        sol._b = inst.w[vals].sum()
        sol.get_obj_val()

    def random_solution(self,sol):
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

    def greedy(self,sol):
        inst = sol.inst
        sortedp = inst.p.argsort()[::-1]
        cumsum = np.cumsum(inst.w[sortedp])
        ind = sortedp[np.argwhere(cumsum <= inst.b).ravel()]
        sol.x[:] = 0
        sol.x[ind] = 1 
        sol.z[:] = -1
        sol.z[:len(ind)] = ind[:]
        sol._b = np.sum(inst.w[ind])
        sol.get_obj_val()

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
        self._b = (self.x * w).sum()
        self.obj = (self.x * p).sum() - M * max(0,self._b-b)
        return self.obj

    def copy_solution(self,sol):
        self.x[:] = sol.x[:]
        self.z[:] = sol.z[:]
        self.obj = sol.obj
        self._b = sol._b

    def print(self):
        self.get_obj_val()
        print(f'obj  : {self.obj:16.2f}')
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

class CInstance():
    def __init__(self,filename):
        self.read_file(filename)

    def read_file(self,filename):
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

class Crono():
    def __init__(self):
        self.start_time = pc()

    def start(self):
        self.reset()

    def get_time(self):
        return (pc() - self.start_time)

    def reset(self):
        self.start_time = pc()

if __name__ == '__main__':
    main()