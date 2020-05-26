import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt


def neighbourhood_weights( network, l, n):

	""" 
	input: index of central node and index of first neighbours
	return the weights of the subgraph of the first neighbours of node n
	"""

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


def random_network(n_nodes,n_links):

	"""
	create an erdos reny graph, given the number of nodes and number of links,
	return its adjagency matrix

	"""

	m = np.zeros((n_nodes,n_nodes))

	max_l = n_nodes*(n_nodes-1)/2
	p = n_links/max_l

	for i in range(n_nodes):
		for j in range(n_nodes):

			if( np.random.uniform(0,1) > p ):
				if(i!=j) :

					r = np.random.uniform(0,1)
					m[i,j] = r
					m[j,i] = r

	return csr_matrix(m)



def scale_free_network(n_nodes): 

	"""
	returns the adjagency matrix of a scale free network
	the network is undirected and weighted

	"""

	m = np.zeros((n_nodes,n_nodes))
	
	G = nx.scale_free_graph(n_nodes) # ritorna diretto
	G = nx.to_undirected(G)
	
	A = nx.adjacency_matrix(G)

	rows    = A.nonzero()[0]
	columns = A.nonzero()[1]

	print(rows.shape,columns.shape)

	for i in range(rows.shape[0]):

			l, k = rows[i], columns[i]

			r  = np.random.uniform(0,1)
			m[l,k] = r
			m[k,l] = r

	return csr_matrix(m)