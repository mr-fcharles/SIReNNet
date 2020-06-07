from numba import njit
from numba import prange
import numpy as np


@njit(parallel=True)
def _link_sampler(links_to_add, dir_sampler,get_richer_step=0.1):
    """
    Functions compiled JIT used to draw the edges to connect. Iternal use
    """

    start_point = np.empty(links_to_add)
    end_point = np.empty(links_to_add)

    for i in prange(links_to_add):

        # extract the prob dist from the dirichlet
        pdist = dir_sampler.sample()

        # use the sampled distrib to draw the node indexes to connect
        #multi_sampler_internal.set_pvals(pdist)

        # extract the two indexes of the graph
        vertex_a = 0
        vertex_b = 0

        while vertex_a == vertex_b:

            draw1 = np.random.multinomial(n=1, pvals=pdist)
            # multinomial draws between 1 and len(self.pdist), our indexes start from 0 and go up to len(self.pdist)-1
            vertex_a = np.argwhere(draw1)[0][0] - 1

            draw2 = np.random.multinomial(n=1, pvals=pdist)
            vertex_b = np.argwhere(draw2)[0][0] - 1

        start_point[i] = vertex_a
        end_point[i] = vertex_b

        # update weights on extracted vertices in order to make new connections more likely
        dir_sampler.alpha[vertex_a] += get_richer_step
        dir_sampler.alpha[vertex_b] += get_richer_step

    return start_point, end_point