import math
import random
import numpy as np
from numpy.linalg import norm

from transport import Transport

def Gillespie(V, F, U, AP, beta, gamma): # aggiungi la variabile sector, i pesi della rete da passare a omega

    C = V
    Pop = norm(V[:-1])
    t = C[-1]

    RHO = F
    N,D = F.shape

    tr_occ = 0 # does transport occurs?

    omega_ngb = Transport(C[:-1], RHO[:,:-1], U) # di F rimuovo l'ultima colonna, il tempo dei nodi
    omega_cnt = -np.sum(omega_ngb,0)

    w1 = beta * C[0] * C[1] / Pop
    w2 = gamma * C[1]
    w3 = norm(omega_cnt,1)

    #print("F: ",F.shape,"w3: ",w3)
   
    W = w1 + w2 + w3


    if(W!=0):

    	dt = -math.log(random.uniform(0.0 , 1.0)) / W
    	t += dt

    	C[-1] = t
    	RHO[:,-1] = t
    
    	r = random.uniform(0.0, 1.0)

    	if r < w1 / W:
            C[0] -= 1 # S -> S-1
            C[1] += 1 # I -> I-1
            return C, RHO, dt, tr_occ

    	if r >= w1/W and r<(w1+w2)/W:
    	    C[1] -= 1 # I -> I-1
    	    C[2] += 1 # R -> R-1
    	    return C, RHO, dt, tr_occ

    	if r>=(w1+w2)/W and r<1:

    		tr_occ = 1

    		for i in range(D-1):
    			C[i] += omega_cnt[i]
    			for j in range(N):
    				RHO[j,i] += omega_ngb[j,i]

    		C[-1] = t
    		RHO[:,-1] = t

    		if ( C>=0 ).all() and ( RHO>=0 ).all():
    			#print("ooooooooo")
    			return C, RHO, dt, tr_occ

    		else:
    			V[-1]   = t
    			F[:,-1] = t
    			#print("uuuuuuuu")
    			#print(V,"\n",F)
    			return V, F, dt, tr_occ

    else:
    	dt = -math.log(random.uniform(0.0 , 1.0)) / AP
    	t+=dt

    	V[-1] = t
    	F[:,-1] = t

    	#print("qqqqqqqq")
    	#print(V,"\n",F)
    	
    	return V, F,  dt, tr_occ

    