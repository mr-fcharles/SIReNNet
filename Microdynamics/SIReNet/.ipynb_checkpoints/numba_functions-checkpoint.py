from numba import njit
from numba import prange
import numpy as np


@njit(parallel=True)
def _link_sampler(links_to_add, dir_sampler, multi_sampler,get_richer_step=0.1):
    """

    :rtype: object
    """
    # start_point=np.empty(links_to_add,np.int64)
    # end_point=np.empty(links_to_add,np.int64)

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
