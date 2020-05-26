import math
import random
import numpy as np
from numpy.linalg import norm

from transport import Transport

"""

                    I N P U T :

  V : state of the central node. The last element is time, the other are the compartments values
  F : state of the first neighbours.
  U : weights of the subgraph
  AP: average population of the network. If no one of the three transitions is possible the node will mantain its state
      and its time will be pushed forward with a an exponential distribution which scale is a mean-field term.


                    B O D Y

  C,RHO :  the idea is to do a copy of the input variable and process them.
           if something goes wrong the function reject the proposed state
           in the spirit of a MCMC evolution, with various check.

 omegas :  those variables represent the transport effect between
           the chosen node in the evolution from the main and its neighbourhood

check #1 : the first if check if the evolution is possible. If C[1], the infected compartment,
           would be zero W, the time scale of the pdf from which the dt is sampled, would be infinite
           and the process meaningless

           if the evolution process is possible the nodes are evolved
           according to a markov jump process. 
           To each transition is associated a weight, namely w1, w2, w3
           transofmed then into a transition probability with p_i = w_i / w1+w2+w3 = w_i / W

           SIR MODEL
     w1  : S --> S-1 , I --> I+1
     w2  : I --> I-1 , R --> R+1
           TRANSPORT EFFECT
     w3  : directly proportional to the total number of people moved by a transport transition

check #2 : even when transitions are possible it does not mean that they are accepted.
           since a negative value of each compartment is meaningless,
           if something in the jump process goes wrong the function will return the input
           otherwise the new state is accepted and given as output


                    O U T P U T:

       C : evolved state of the centre 
     RHO : same for the neighbours

           """


beta  = 0.51
gamma = 0.25

def Gillespie(V, F, U, AP): # aggiungi la variabile sector, i pesi della rete da passare a omega

    C = V
    Pop = norm(V[:-1],1)
    
    t = C[-1]

    RHO = F
    N,D = F.shape

    omega_ngb = Transport(V[:-1], F[:,:-1], U) # di F rimuovo l'ultima colonna, il tempo dei nodi
    omega_cnt = -np.sum(omega_ngb,0)

    w1 = beta * C[0] * C[1] / Pop
    w2 = gamma * C[1]
    w3 = norm(omega_cnt,1)
   
    W = w1 + w2 + w3

    dt = -math.log(random.uniform(0.0 , 1.0)) / W
    t += dt

    C[-1] = t
    RHO[:,-1] = t

    r = random.uniform(0.0, 1.0)

    if r < w1 / W:
        C[0] -= 1 # S -> S-1 
        C[1] += 1 # I -> I-1
        return C, RHO

    if r >= w1/W and r<(w1+w2)/W:
    	C[1] -= 1 # I -> I-1
    	C[2] += 1 # R -> R-1
    	return C, RHO

    if r>=(w1+w2)/W and r<1:
        
        C[:-1]     += omega_cnt
        RHO[:,:-1] += omega_ngb

        if ( C>=0 ).all() and ( RHO>=0 ).all():
            return C, RHO
        else:
            V[-1]   = t
            F[:,-1] = t
            return V, F

    
