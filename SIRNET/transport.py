import random
import numpy as np
from numpy.random import binomial as binomial
from numpy.random import multinomial as multinomial 

import math

"""

INPUT
    
      V ---> mat(1xD) stato del nodo centrale del subgrafo

      F ---> mat(NxD) numero di primi vicini
             è lo stato del nodo da evolvere più quello del suo viciniato

      M ---> mat(N) è il nodo dei pesi della rete, 
             dai quali si possono calcolare le probabilità

OUTPUT

	  OM --> mat(N-1xD) che ritorna il tasso di trasporto 
	         di ciascun compartimento per ciascun nodo del vicinato
	         l'evoluzine di ogni compartimento del nodo centrale viene fatta
	         nell'evoluzione principale, OM(centr,X) = -sum_N( OM(X)) 
	         in modo tale che si conservi ad ogni transizione il numero totale
	         "di particelle" per ogni vicinato


"""


def Transport( V, F, W ): # Transport(m, neibgourhood, pesi neib)

	# RIMUOVI IL QUARTO GRADO DI LIBERTA', IL TEMPO

	N,D = F.shape

	OM = np.zeros((N,D)) 

	prob = W/np.sum(W)

	for i in range(D):

		C = V[i]
		xi_in = multinomial(C,prob)

		for j in range(N):

			xi_out = binomial(F[j,i],prob[j])

			OM[j,i] = xi_in[j] - xi_out


	return OM


	
"""
	for i in range(D): # esegui un'operzaione di trasporto per ogni compartimento

		C      = V[i] # grandezza del compartimento corrente del nodo centrale 
		xi_out = multinomial(C,prob) # occhio a cosa estrai, nel senso quale è del nodo quali sono gli altir?

		for j in range(N): # scelto il grado di libertà applichi il trasporto ad ogni nodo

			T = F[j,i] # compartimento i del nodo j

			print(N,prob[j])	
			# il problema è qui! F è per 4 nodi ma p è per 4-1!
			xi_in   = binomial(T,prob[j])

			print(" aaa ",xi_out[i]," ooo ",xi_in," ")
			OM[i,j] = xi_out[i] - xi_in
"""

	#return OM