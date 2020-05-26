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

from timeseries import TSPRCSS

import timeit


print(" ========= PARAMETERS ========= ")

"""
maximum simulation time, eg number of days
number of nodes into the network, eg number of cities
number of compartments into the model
mean size of the city population
"""

T_SIMULATION = 60
communities  = 100
DOF          = 4
MeanP        = 500


print(" ========= INITIALIZATION ========= ")

"""
random initialization of the system. 
This section outputs an array with the initial condition of each node in the newtork
nodes are chosen randomly anc stored into the epicenter array 
but one can improve it for example choosing an initial node and then perform a random walk on the graph

see initialization.py
"""

epicenter = np.unique( np.random.randint(0,communities,int(0.1*communities) ))

stat_parameters = np.array([MeanP,MeanP/2])
Z0              = RND_State0(epicenter, communities, DOF, stat_parameters)



print("========= DEMOGRAPHICS =========")

"""
some demographic stuff
demo = initial total population of each node
tot_pop total population of the graph
then the dynamical system is initialized with the varaible F 
avg_population will be used in the evolution as an "escape parameter" in the markov process

see dynamics.py
"""

demo    = np.sum(Z0,1).astype(int)
tot_pop = np.sum(demo)
avg_pop = tot_pop/communities

print("Total initial populaiton: ", tot_pop)

F = System(Z0)



print(" ========= NET =========")

"""
this section builds the network matrix containing the weights
they are initialized randomly with uniform distribution 
and then normalized in the tranport.py file in order to build the transport process
network is a sparse scipy matrix, the networkx stuff is just for visualization

see mat.py
"""

#network = random_network(communities,6000)
network = scale_free_network(communities)

G = nx.Graph(network)

nx.draw(G)
plt.show()

print(" ========= RUN ========= ")

"""

proper simulation part.
there are two check for the lenght of the simulation.
One is that the least evolved node time reaches the minimum amount
the second is over the number of iterations

STEP 1: the algorithm choose the node(s) to evolve
STEP 2: extract from the adjagency matrix the weights of the subgraph
        and the state of the neighbourhood
STEP 3: perform the evolution
STEP 3: save the evolution

"""

start = timeit.default_timer()
progress = 0

while np.mean( F.all_local_times() ) < T_SIMULATION:

	if progress%4000==0: 
		print("ITER: ", progress)
	if progress > 2e5:
		break


	# select the nodes to evolve

	m           = np.random.randint(0,communities)
	ngb         = first_neighbours(network,m) # array con l'etichetta dei primi vicini
	
	cnt_state = F.instant_state(m)
	ngb_state = F.instant_neibgourhood( ngb ) # rifst row il node m, other are its first neighbours
	ngb_weights = neighbourhood_weights( network, ngb, m)

	c, p = Gillespie(  cnt_state, ngb_state, ngb_weights, avg_pop )

	F.evolve_node(m,c)

	i=0 
	for n in ngb:

		F.evolve_node(n,p[i,:])
		i+=1

	progress+=1

 

stop = timeit.default_timer( )
print("Simulation runtime: ", stop - start)		


print(" ========= PLOT ========= ")


for i in range(5):
	
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

TSPRCSS(F, tot_pop, T_SIMULATION)

