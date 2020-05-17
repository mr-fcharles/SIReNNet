import random
import numpy as np
import networkx as nx
import math
#import scipy
import matplotlib.pyplot as plt

from gilevo import Gillespie
from initialization import RND_State0
from dynamics import System

from mat import random_network
from mat import scale_free_network
from mat import neighbourhood_weights
from mat import first_neighbours

import timeit

start = timeit.default_timer()

print(" ========= PARAMETERS ========= ")

T_SIMULATION = 400

communities = 4500 # number of nodes, links, probabilitiy
DOF = 4 # degrees of freedom

MeanP = 5000 # mean city size

#infected_cities0 = 10 
#np.random.randint(int(0.10*Communities),int(0.15*Communities))




print(" ========= INITIALIZATION ========= ")
# metti un random walk per avere diffusione iniziale di asintomatici

epicenter = np.unique( np.random.randint(0,communities,30 ))


print("Epicenter: \n" , epicenter)

stat_parameters = np.array([MeanP,MeanP/2])

Z0 = RND_State0(epicenter, communities, DOF, stat_parameters)




print("========= DEMOGRAPHICS =========")

demo = np.sum(Z0,1).astype(int)

tot_pop = np.sum(demo)
print("Total initial populaiton: ", tot_pop)

avg_pop = tot_pop/communities

F = System(Z0)

#print("Epicenter initial State: ")
#print(Z0)




print(" ========= NET =========")

#network = random_network(communities,0.80)

network = scale_free_network(communities)

G = nx.Graph(network)
print(G.number_of_edges())
#nx.draw(G)
#plt.show()

print(" ========= RUN ========= ")


progress = 0

beta  = 0.51
gamma = 0.32

population = np.sum(Z0,0)[:-1]
# BO ALLORA DEFINISCI POPULATION CON SYSTEM, 
#CIOE' IMPLEMENTI UN METODO CHE RITORNI LA SOMMA SUI COMPARTIMANTI E IL TEMPO
err = 0
cnt_state = np.zeros(4)
evol = np.zeros(4)
delta_c = np.zeros(4)

while min( F.all_local_times() ) < T_SIMULATION:

	if progress%4000==0: 
		print("ITER: ", progress)
		#err = sum(population[-8:-4]) - sum(population[-4:-1] )
		#if(err!=0):
			#print("ERR:  ", err  )
	if progress > 2e5:
		break


	# select the nodes to evolve

	m           = np.random.randint(0,communities)
	ngb         = first_neighbours(network,m) # array con l'etichetta dei primi vicini
	ngb_weights = neighbourhood_weights( network, ngb, m)

	# extract nodes state, one array for the centre one for first ngb nodes


	cnt_state = F.instant_state(m)
	ngb_state = F.instant_neibgourhood( ngb ) # rifst row il node m, other are its first neighbours
	
	#print("Node: ",m)
	c, p, dt, tr_happens = Gillespie(  cnt_state, ngb_state, ngb_weights, avg_pop, beta, gamma )
	#fai un attimo le somme per vedere se c'è conservazione

	#delta_c = c - cnt_state
	#delta_p = np.sum(p,0) - np.sum(ngb_state,0)

	#print(c,cnt_state)
	#print(delta_p)

	#evol = delta_p + delta_c
	#evol[-1]+=dt

	#print(evol)
	#population = np.append( population, population[-4:] + c -q )
	

	F.evolve_node(m,c)
	i=0 
	for n in ngb:

		F.evolve_node(n,p[i,:])
		i+=1

	population = np.append( population, F.instant_population() )


	"""
	new_cnt_state = F.instant_state(m)
	if(tr_happens==0):
		delta_c[:-1] = (new_cnt_state - cnt_state)[:-1]
		delta_c[-1] += dt
		population = np.append( population, population[-4:] + delta_c )
	"""

	progress+=1


stop = timeit.default_timer()

print('Time: ', stop - start)	

# FAI IL PLOT ANCHE DEI SINGOLI NODI

print(" ========= PLOT ========= ")

#print(population.shape)
#l = int(population.shape[0]/3)
#print(l)

#population = population.reshape(l,3)
	
SIR_State = population


S    = SIR_State[0::3]
I    = SIR_State[1::3]
R    = SIR_State[2::3]

print(S.shape,I.shape,R.shape)

l = S.shape[0]
print(l)

time = np.linspace(0,T_SIMULATION,l)
print(time.shape)

N = tot_pop # correggi, ci va quella del nodo!

plt.plot(time,S/N, label="S(t)")
plt.plot(time,I/N, label="I(t)")
plt.plot(time,R/N, label="R(t)")

plt.legend()
plt.xlabel("Time")
plt.ylabel("Percentage")
plt.show()


for i in range(communities):
	
	SIR_State = F.trajectory(i)
	print(SIR_State.shape)
	time = SIR_State[:,3]
	S    = SIR_State[:,0]
	I    = SIR_State[:,1]
	R    = SIR_State[:,2]

	N = demo[i] # correggi, ci va quella del nodo!

	plt.plot(time,S/N, label="S(t)")
	plt.plot(time,I/N, label="I(t)")
	plt.plot(time,R/N, label="R(t)")

	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Percentage")
	plt.show()

