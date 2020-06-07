#old versions of some of the methods and functions of the project

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

    # initialize the dirichlet sampler and draw the first distrib
    dir_sampler = Dirichlet(size=self.pop_size, alpha=1)
    pdist = dir_sampler.sample()

    # initialiaze the multinomial sampler and set the drawn distrib
    multi_sampler = Multinomial(size=1, pdist=pdist)
    # multi_sampler = Multinomial_with_randomization(size=1, pdist=pdist,randomization=0)

    added = 0

    # initialize the vectors in which we will store the nodes to connect
    edge_a = np.empty(links, dtype=np.int32)
    edge_b = np.empty(links, dtype=np.int32)

    while (added < links):

        if (added == 0):

            edge_a, edge_b = _link_sampler(links - added, dir_sampler, multi_sampler, get_richer_step)

            edge_a = edge_a.astype(np.int32)
            edge_b = edge_b.astype(np.int32)

        # apprently some times numba classes incur in numerical errors
        # the code below takes care of redrawing instable observations

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

    # fianlly we insert in the adjacency matrix the obtained links

    for j in range(links):
        # print(edge_a[j],edge_b[j])
        row = edge_a[j]  # - 1
        column = edge_b[j]  # - 1

        self.adjacency[row, column] = True
        self.adjacency[column, row] = True

    # we remove self loops
    self.adjacency.setdiag(0)




#################################################################
from SIReNet.numba_samplers import Multinomial

@njit(parallel=True)
def _link_sampler(links_to_add, dir_sampler, multi_sampler,get_richer_step=0.1):
    """
    Functions compiled JIT used to draw the edges to connect. Iternal use
    """


    start_point = np.empty(links_to_add)
    end_point = np.empty(links_to_add)

    for i in prange(links_to_add):

        # extract the prob dist from the dirichlet
        pdist = dir_sampler.sample()

        # use the sampled distrib to draw the node indexes to connect
        multi_sampler.set_pvals(pdist)

        # extract the two indeces of the graph
        vertex_a = 0
        vertex_b = 0

        while vertex_a == vertex_b:
            vertex_a, vertex_b = multi_sampler.sample()

        start_point[i] = vertex_a
        end_point[i] = vertex_b

        # update weights on extracted vertices in order to make new connections more likely
        dir_sampler.update((vertex_a, vertex_b), get_richer_step)

    return start_point, end_point