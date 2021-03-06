import numpy as np
import warnings
from scipy.sparse import dok_matrix

from SIReNet.numba_samplers import Dirichlet
from SIReNet.numba_samplers import Multinomial
from SIReNet.experimental import Multinomial_with_randomization
from SIReNet.numba_functions import _link_sampler

class SparseGraph(object):
    '''

    '''

    def __init__(self, pop_size=1000, node_name='Paperopolis'):

        # object identifiers
        self.node_name = node_name

        # graph strucutre
        self.pop_size = pop_size
        self.adjacency = dok_matrix((pop_size, pop_size), dtype=np.bool)
        self.common_neighbors = None

    #########################################################

    def create_families(self, lambda_mean=2):
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
            
            multi_sampler = Multinomial(size=1, pdist=pdist)
            #multi_sampler = Multinomial_with_randomization(pdist=pdist,randomization=0)

            _link_sampler(1, dir_sampler, multi_sampler)

    #######################################################

    def add_links(self, get_richer_step=0.1, links_to_add='default'):
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

            dir_sampler = Dirichlet(size=self.pop_size, alpha=1)
            pdist = dir_sampler.sample()
            
            
            multi_sampler = Multinomial(size=1, pdist=pdist)
            #multi_sampler = Multinomial_with_randomization(pdist=pdist,randomization=0)

            added = 0

            edge_a = np.empty(links, dtype=np.int32)
            edge_b = np.empty(links, dtype=np.int32)

            while (added < links):

                if (added == 0):

                    edge_a, edge_b = _link_sampler(links - added, dir_sampler, multi_sampler,get_richer_step)

                    edge_a = edge_a.astype(np.int32)
                    edge_b = edge_b.astype(np.int32)

                else:

                    warnings.warn('Numerical error occurred, remaning links {}'.format(links - added))

                    # print('Numerical occurred, remaining links:',links-added)

                    temp_a, temp_b = _link_sampler(links - added, dir_sampler, multi_sampler)

                    edge_a = np.append(edge_a, temp_a)
                    edge_b = np.append(edge_b, temp_b)

                    added += len(temp_a)

                # first filter
                filter1 = np.where(np.abs(edge_a.astype(np.int64)) > self.pop_size, False, True)

                edge_a = edge_a[filter1]
                edge_b = edge_b[filter1]

                # second filter
                filter2 = np.where(np.abs(edge_b.astype(np.int64)) > self.pop_size, False, True)

                edge_a = edge_a[filter2]
                edge_b = edge_b[filter2]

                if (added == 0):
                    added += len(edge_a)

            for j in range(links):
                # print(edge_a[j],edge_b[j])
                row = edge_a[j] #- 1
                column = edge_b[j] #- 1

                self.adjacency[row, column] = True
                self.adjacency[column, row] = True

            self.adjacency.setdiag(0)


    #############################################################################

    def compute_common_neighbors(self):

        try:

            adjac_temp = self.adjacency.astype(np.float32)
            adjac_temp.setdiag(0.5)

            self.common_neighbors = adjac_temp.dot(adjac_temp)
            self.common_neighbors = self.common_neighbors.todok().astype(np.int16)
            self.common_neighbors.setdiag(1)

            return (self.common_neighbors)

        except:

            print('stocazzo')



    ############################################################################
    def adjacency_tocsr(self):

        try:
            self.adjacency = self.adjacency.tocsr()

        except:

            print('Adjacency matrix not initialized')

