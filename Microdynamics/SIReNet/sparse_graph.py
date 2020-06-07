import numpy as np
import warnings
from scipy.sparse import dok_matrix

from SIReNet.numba_samplers import Dirichlet
from SIReNet.numba_functions import _link_sampler
from SIReNet.utils import _adj_writer

from joblib import Parallel,delayed



class SparseGraph(object):
    '''
    Base class used to represent and generate a graph through sparse matrices

    Attributes
    ___________

    - node_name: str
        the name that we want to give to this graph (ex.: the city we are modelling)

    - pop_size: int > 0
        the number of nodes in the graph

    - adjacency_matrix: scipy.sparse.dok_matrix
        models the connections between nodes (should be symmetric since we are considering undirected graphs)

    - common neighbors: scipy.sparse.csr_matrix
        modfied version of common neighbors matrix


    Methods
    __________

    - create_families(lambda_mean: float>0)
        Creates the basic "family cliques"

    - sampler_initializer()
        Initializes jit compiled functions

    - add_links(get_richer_step: float > 0, links_to_add: int >0 or str )
        Adds link to adjacency matrix with preferential attachment

    - compute_common_neighbors()
        Compute commons neighbors

    - adjacency_to_csr()
        Converts adjecncy to csr format


    '''
    def __init__(self, pop_size=1000, node_name='Paperopolis'):

        # object identifiers
        self.node_name = node_name

        # graph strucutre
        self.pop_size = pop_size
        self.adjacency = dok_matrix((pop_size, pop_size), dtype=np.bool)
        self.common_neighbors = None

    #########################################################

    def create_families(self, lambda_mean=2.4):
        '''
        This method is used to populate the sparse adjacency matrix of the
        graph with families (i.e. many small cliques)

        :param lambda_mean : (real number > 0) average size of the families

        :returns: Modifies the adjacency matrix saved in self.adjacency
        '''

        #We initialized a counter at zero that has to be increased of the family dimension
        i = 0

        while i < self.pop_size:

            # we start by drawing the dimension of the population from a poisson distribution
            family_size = max(1, np.random.poisson(lam=lambda_mean))

            try:

                 # we use a double cycle to connect each member of the family to everyone else in the family
                for k in range(family_size):

                    for j in range(family_size):
                        self.adjacency[i + k, i + j] = 1

                i = i + family_size

            # except used in order to add the last family
            except:

                 remainder_size = self.pop_size - i

                 for k in range(remainder_size):

                     for j in range(remainder_size):
                         self.adjacency[i + k, i + j] = 1

                 i = self.pop_size

        #remove self loops
        self.adjacency.setdiag(0)

    #######################################################

    def sampler_initializer(self):
            '''
            Numba functions are compiled just in time the first time we call
            them. This function initializes both numba classes and the link
            sampler in order not to incourr into the computational overhead
            when we need to add a massive number of edges

            :return: Initializes numba methods and functions
            '''


            # links
            dir_sampler = Dirichlet(size=self.pop_size, alpha=1)
            pdist = dir_sampler.sample()

            _link_sampler(1, dir_sampler)

    #######################################################


    def add_links(self, get_richer_step=0.1, links_to_add='default',n_jobs=None):
        '''
        Method used to add randomly to the graph links with preferential attachment

        :param get_richer_step: (real number > 0)  The higher is this parameter, the heavier
        are the degree distribution tails
        :param links_to_add: (integer > 0) Number of edges to add to the graph with this procedure
        :return: Adds drawn link to self.adjacency
        '''

        if (links_to_add == 'default'):
            links = self.pop_size * 3

        else:
            links = links_to_add

        # initialize the dirichlet sampler and draw the first distrib
        dir_sampler = Dirichlet(size=self.pop_size, alpha=1)

        edge_a, edge_b = _link_sampler(links, dir_sampler, get_richer_step)

        if(n_jobs is None):
        # fianally we insert in the adjacency matrix the obtained links

            for j in range(links):
                # print(edge_a[j],edge_b[j])
                row = edge_a[j]  # - 1
                column = edge_b[j]  # - 1

                self.adjacency[row, column] = True
                self.adjacency[column, row] = True

        # todo not very fast, should undestand why
        else:
            Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_adj_writer)(self.adjacency, in_node, out_node) for in_node in edge_a for out_node in edge_b)

        # we remove self loops
        self.adjacency.setdiag(0)

    #############################################################################

    def compute_common_neighbors(self):

        '''
        Method used to compute the modified common neighbor matrix. We can compute
        the common neighbors matrix by multiplying the adjacency matrix by itself.
        However, since later we will use this matrix to "weight" the infection probability
        by an element wise moltiplication of adjacency matrix and common negihbors matrix
        we need to slightly modify the latter: nodes that are connected but do not share
        any nieghbors shoud be related by a 1 in common neighobrs matrix. This is achieved
        by setting the diagonal of the adjacency matrix to 0.5.

        :return:
        The modified common neighbors matrix is computed and stored in the attribute
         self.common_neighbors
        '''


        adjac_temp = self.adjacency.astype(np.float32)
        adjac_temp.setdiag(0.5)

        self.common_neighbors = adjac_temp.dot(adjac_temp)
        self.common_neighbors = self.common_neighbors.todok().astype(np.int16)

        #common neighbors of a node itself is equivalent to degree, we reset this to 1
        self.common_neighbors.setdiag(1)

        return (self.common_neighbors)





    ############################################################################
    def adjacency_tocsr(self):

        '''
        The adjacency matrix is initialized as a dok sparse matrix. If needed
        this method converts the latter into the csr format

        :return:
        Converts self.adjacency matrix to csr format
        '''

        try:
            self.adjacency = self.adjacency.tocsr()

        except:

            print('Adjacency matrix not initialized')

