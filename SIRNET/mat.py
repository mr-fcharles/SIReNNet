import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt


def neighbourhood_weights( network, l, n):

	s = l.shape[0]

	w = np.zeros(s)

	for i in range(s):
		w[i] = network[ l[i], n]

	return w


# returns indexes of the first neighbours of node n
# if n is linked to nodes k and l result will be an array
# like (k,l)

def first_neighbours(network, n):

	return np.array(network.getrow(n).nonzero())[1,:]

# crea un network random

def random_network(n_nodes,p):

	m = np.zeros((n_nodes,n_nodes))
	p = 0.35

# se il grafo è indiretto ci si può tenere i link
# solo della parte triangolare inferiore, se si vuole iterare sulle colonne

	for i in range(n_nodes):
		for j in range(n_nodes):

			if( np.random.uniform(0,1) < p ):
				if(i!=j) :

					r = np.random.uniform(0,1)
					m[i,j] = r
					m[j,i] = r

# buttalo in una matrice sparsa

	return csr_matrix(m)



def scale_free_network(n_nodes):

	m = np.zeros((n_nodes,n_nodes))
	
	G = nx.scale_free_graph(n_nodes) # ritorna diretto
	G = nx.to_undirected(G)
	
	A = nx.adjacency_matrix(G)

	rows    = A.nonzero()[0]
	columns = A.nonzero()[1]

	print(rows.shape,columns.shape)

	for i in range(rows.shape[0]):

			l, k = rows[i], columns[i]

			nn_a = first_neighbours(A,l).shape[0]
			nn_b = first_neighbours(A,k).shape[0]
			nn = abs(nn_a + nn_b )

			r  = np.random.exponential( scale=1/nn)
			m[l,k] = r
			m[k,l] = r

# buttalo in una matrice sparsa

	return csr_matrix(m)