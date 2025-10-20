import os
import sys
import numpy as np
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY
from time import perf_counter as pc

np.random.seed(5000)

def main():
    assert len(sys.argv) > 1, 'please,provide a data file'
    inst = CInstance(sys.argv[1])
    #inst.print()

    # Soluções iniciais para os testes
    sol_rnd = CSolution(inst)
    sol_greedy = CSolution(inst)

    constr = CConstructor()
    
    print('construction phase')   
    
    # Usando o melhor método construtivo do primeiro trabalho prático
    # Assumindo o método guloso ('greedy') como o "melhor método construtivo"
    print('greedy ')
    constr.greedy(sol_greedy)
    #sol_greedy.print()

    # Usando a construção 'random 2' como a outra opção no código original
    print('random 2')
    constr.random_solution2(sol_rnd)
    #sol_rnd.print()
    
    ls = CBuscaLocal()

    print('\n--- VND Experiments (starting from Greedy Solution) ---')
    # 1. VND: 1-Opt seguido de 2-Opt (Vizinhanças: N1=1-Opt, N2=2-Opt)
    sol12 = CSolution(inst)
    sol12.copy_solution(sol_greedy)
    print('VND First Improvement (1-Opt -> 2-Opt)')   
    ls.vnd(sol12, strategy='first', neighborhood_order='12')
    sol12.print()
    
    sol12_best = CSolution(inst)
    sol12_best.copy_solution(sol_greedy)
    print('VND Best Improvement (1-Opt -> 2-Opt)')   
    ls.vnd(sol12_best, strategy='best', neighborhood_order='12')
    sol12_best.print()

    # 2. VND: 2-Opt seguido de 1-Opt (Vizinhanças: N1=2-Opt, N2=1-Opt)
    sol21 = CSolution(inst)
    sol21.copy_solution(sol_greedy)
    print('VND First Improvement (2-Opt -> 1-Opt)')   
    ls.vnd(sol21, strategy='first', neighborhood_order='21')
    sol21.print()
    
    sol21_best = CSolution(inst)
    sol21_best.copy_solution(sol_greedy)
    print('VND Best Improvement (2-Opt -> 1-Opt)')   
    ls.vnd(sol21_best, strategy='best', neighborhood_order='21')
    sol21_best.print()
    
    # 3. Random VND (RVND)
    sol_rvnd = CSolution(inst)
    sol_rvnd.copy_solution(sol_greedy)
    print('RVND Best Improvement')   
    ls.rvnd(sol_rvnd, strategy='best') # O RVND tipicamente usa Best Improvement
    sol_rvnd.print()

    print('\n--- VNS/Smart VNS Experiments (starting from Greedy Solution) ---')
    MAX_TIME = 3 # Tempo máximo para a busca
    H_MAX = 5 # Número máximo de iterações do VNS/Smart VNS
    P_MAX = 3 # Número máximo de perturbações na mesma vizinhança para o Smart VNS

    # VNS Básico com VND First Improvement (1-Opt -> 2-Opt)
    sol_vns_f12 = CSolution(inst)
    sol_vns_f12.copy_solution(sol_greedy)
    print(f'VNS Básico (Tempo: {MAX_TIME}s) com VND First (1-Opt -> 2-Opt)')
    ls.vns(sol_vns_f12, max_time=MAX_TIME, strategy='first', neighborhood_order='12')
    sol_vns_f12.print()

    # VNS Básico com VND Best Improvement (2-Opt -> 1-Opt)
    sol_vns_b21 = CSolution(inst)
    sol_vns_b21.copy_solution(sol_greedy)
    print(f'VNS Básico (Tempo: {MAX_TIME}s) com VND Best (2-Opt -> 1-Opt)')
    ls.vns(sol_vns_b21, max_time=MAX_TIME, strategy='best', neighborhood_order='21')
    sol_vns_b21.print()

    # Smart VNS com VND Best Improvement (1-Opt -> 2-Opt)
    sol_svns_b12 = CSolution(inst)
    sol_svns_b12.copy_solution(sol_greedy)
    print(f'Smart VNS (H_max: {H_MAX}, P_max: {P_MAX}) com VND Best (1-Opt -> 2-Opt)')
    ls.svns(sol_svns_b12, h_max=H_MAX, p_max=P_MAX, strategy='best', neighborhood_order='12')
    sol_svns_b12.print()
    
    # Smart VNS com VND First Improvement (2-Opt -> 1-Opt)
    sol_svns_f21 = CSolution(inst)
    sol_svns_f21.copy_solution(sol_greedy)
    print(f'Smart VNS (H_max: {H_MAX}, P_max: {P_MAX}) com VND First (2-Opt -> 1-Opt)')
    ls.svns(sol_svns_f21, h_max=H_MAX, p_max=P_MAX, strategy='first', neighborhood_order='21')
    sol_svns_f21.print()


    mod = CModel(inst)
    mod.run()
     

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

class CBuscaLocal():
    def __init__(self):
        # Mapeamento das funções de vizinhança
        self.neighborhoods = {
            '1': self.swap_one_bit, # First Improvement
            '2': self.swap_two_bits, # First Improvement
            '1b': self.swap_one_bit_best_improvement, # Best Improvement
            '2b': self.swap_two_bits_best_improvement # Best Improvement
        }

    def rvnd(self,sol, strategy='best'):
        # O RVND (Randomized VND) deve ser uma função de busca local
        # que usa uma ordem aleatória das vizinhanças e reinicia
        # com a primeira vizinhança (aleatória) após uma melhora.
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        
        # Estruturas de vizinhança: '1' e '2'
        N_list = ['1b', '2b'] if strategy == 'best' else ['1', '2']
        n_max = len(N_list)
        
        # A ordem é determinada aleatoriamente a cada iteração de melhora
        
        k = 1
        while (k <= n_max):
            # No RVND, a ordem é aleatória. Vamos embaralhar no início e
            # a cada vez que a solução melhorar (k volta a 1)
            if k == 1:
                randn = np.arange(n_max)
                np.random.shuffle(randn)

            # Usa a vizinhança da ordem embaralhada
            viz_key = N_list[randn[k-1]]
            
            # Aplica a busca local na vizinhança viz_key
            # Note: as funções 'swap_one_bit' e 'swap_two_bits' (e suas versões 'best')
            # já implementam o loop de busca local até o ótimo local para aquela vizinhança.
            self.neighborhoods[viz_key](sol)
            
            # Lógica de mudança de vizinhança do VND (Sequential Neighborhood Change)
            if sol.obj > solstar.obj:
               solstar.copy_solution(sol)
               k = 1 # Reinicia com a primeira vizinhança aleatória
            else:
               k += 1 # Passa para a próxima vizinhança aleatória

        sol.copy_solution(solstar) # Retorna a melhor solução encontrada
        

    def vnd(self,sol,strategy='best', neighborhood_order='12'):
        # neighborhood_order: '12' para 1-Opt, 2-Opt ou '21' para 2-Opt, 1-Opt
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)

        # Determina a ordem das vizinhanças (1=swap_one_bit, 2=swap_two_bits)
        if neighborhood_order == '12':
            N_list = ['1', '2']
        elif neighborhood_order == '21':
            N_list = ['2', '1']
        else: # Default
            N_list = ['1', '2']
            
        if strategy == 'best':
            N_list = [n + 'b' for n in N_list] # '1b', '2b'
            
        n_max = len(N_list)
        
        k = 1
        while (k <= n_max):
            viz_key = N_list[k-1]
            
            # Busca local na vizinhança N_k
            self.neighborhoods[viz_key](sol)

            # Lógica do VND (Sequential Neighborhood Change)
            if sol.obj > solstar.obj:
               solstar.copy_solution(sol)
               k = 1 # reinicia com a primeira vizinhança
            else:
               k += 1 # passa para a próxima vizinhança

        sol.copy_solution(solstar) # Retorna a melhor solução encontrada

    
    def vns(self,sol,max_time,strategy='best', neighborhood_order='12'):
        crono = Crono()
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        
        # Determina a ordem das vizinhanças para shaking
        if neighborhood_order == '12':
            N_shake = [self.random_swap_one_bit, self.random_swap_two_bits]
        elif neighborhood_order == '21':
            N_shake = [self.random_swap_two_bits, self.random_swap_one_bit]
        else:
            N_shake = [self.random_swap_one_bit, self.random_swap_two_bits]
            
        n_max = len(N_shake)

        while crono.get_time() < max_time:
            k = 1
            while k <= n_max:
                # 1. Shaking (Agitação/Perturbação)
                # Note: O VNS Básico (Algoritmo 3.0.2) usa a k-ésima vizinhança para perturbar.
                # O VNS Básico no código fornecido usa random_swap_one_bit para h=1 e random_swap_two_bits para h=2.
                # A função shake é N_shake[k-1]. A implementação random_swap_... já perturba.
                N_shake[k-1](sol) # Perturba S na vizinhança N_k

                # 2. Busca Local (usando VND)
                sol_linha = CSolution(sol.inst)
                sol_linha.copy_solution(sol) # Faz uma cópia para a busca local
                # Usa o VND com a estratégia e ordem de vizinhança especificadas
                self.vnd(sol_linha, strategy=strategy, neighborhood_order=neighborhood_order)

                # 3. Move or Not
                if sol_linha.obj > solstar.obj:
                   solstar.copy_solution(sol_linha)
                   sol.copy_solution(solstar) # Atualiza a solução atual S
                   k = 1 # Reinicia a perturbação com a primeira vizinhança
                else:
                   k += 1 # Aumenta o nível de perturbação (próxima vizinhança)
                   
        sol.copy_solution(solstar)


    def svns(self,sol,h_max,p_max,strategy='best', neighborhood_order='12'):
        # Implementação do Smart Variable Neighborhood Search (Algoritmo 3.0.3)
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)
        
        # Determina a ordem das vizinhanças para shaking
        if neighborhood_order == '12':
            N_shake = [self.random_swap_one_bit, self.random_swap_two_bits]
        elif neighborhood_order == '21':
            N_shake = [self.random_swap_two_bits, self.random_swap_one_bit]
        else:
            N_shake = [self.random_swap_one_bit, self.random_swap_two_bits]
            
        n_max = len(N_shake)
        
        h = 0
        while h <= h_max:
            h += 1 # 4. h <- h + 1
            k = 1 # 5. k <- 1
            p = 1 # 6. p <- 1
            
            while k <= n_max: # 7. while k <= n do
                # Shaking: O Smart VNS (Algoritmo 3.0.3) perturba com intensidade 'p'
                # A implementação do shaking (random_swap_...) no código original
                # não suporta intensidade 'p'. Para simular, vamos chamar a função 'p' vezes.
                sol_temp_shake = CSolution(sol.inst)
                sol_temp_shake.copy_solution(solstar) # Começa do S* ou S? O algoritmo 3.0.3 usa 'S' no shaking, mas 'S*' na busca.
                                                      # Usaremos a solução 'S*' como ponto de partida (incumbent), que é o usual em VNS.
                # Como o algoritmo 3.0.3 usa $S'$ Agitação $(S,N_k,p)$, vamos usar $S^{*}$ como S.
                # No VNS Básico é $S'$ Agitação $(S,N_k)$ e a busca é no $S'$ gerado.
                # Assumo que S é a solução incumbente $S^{*}$
                sol_shaked = CSolution(sol.inst)
                sol_shaked.copy_solution(solstar)
                
                # Perturba p vezes na vizinhança N_k
                for _ in range(p):
                    N_shake[k-1](sol_shaked) # S' <- Agitação (S, N_k, p)
                
                # Busca Local (usando VND/SVND)
                # S' <- SVND(S', N)
                self.vnd(sol_shaked, strategy=strategy, neighborhood_order=neighborhood_order)
                
                sol_linha = sol_shaked # sol_linha é o resultado da busca local
                
                # Move or Not
                if sol_linha.obj > solstar.obj:
                    solstar.copy_solution(sol_linha) # S* <- S'
                    k = 1 # k <- 1
                    p = 1 # p <- 1
                else:
                    if p == p_max: # if p = p_max then
                        k += 1 # k <- k + 1
                        p = 1  # p <- 1
                    else: # else
                        p += 1 # p <- p + 1
        
        sol.copy_solution(solstar)

    # Métodos de Shaking (perturbation)
    def random_swap_one_bit(self,sol):
        # ... (Mantido como está no código original: perturba aleatoriamente em N1)
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
        # ... (Mantido como está no código original: perturba aleatoriamente em N2)
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
    
    # Métodos de Busca Local (First Improvement)
    def swap_one_bit(self,sol):
        # ... (Mantido como está no código original)
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
                     break # First Improvement: para no primeiro movimento melhorante
   
    def swap_two_bits(self,sol):
        # ... (Mantido como está no código original)
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
                         break # First Improvement: para no primeiro movimento melhorante
                      h2 += 1
                  if found_improvement:
                      break
                  h1 += 1
              if not found_improvement:
                  delta = -1 # Garante a saída do loop externo


    # Métodos de Busca Local (Best Improvement)
    def swap_one_bit_best_improvement(self,sol):
        # ... (Mantido como está no código original)
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
        # ... (Mantido como está no código original)
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
    # ... (Mantido como está no código original)
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
    # ... (Mantido como está no código original)
    def __init__(self,inst):
        self.inst = inst
        self.x = np.zeros(inst.n)
        self.z = np.full(inst.n,-1)
        self.obj = 0.0
        self._b = inst.b

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
    # ... (Mantido como está no código original)
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

class Crono():
    # ... (Mantido como está no código original)
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