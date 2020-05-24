from numba.experimental import jitclass
from numba import int64, float64
from numba import prange
import numpy as np


############################ DIRICHLET ########################

spec = [
    ('size', int64),
    ('alpha', float64[:])
]

@jitclass(spec)
class Dirichlet(object):

    def __init__(self, size=10, alpha=0.1):
        self.size = size
        self.alpha = np.ones(self.size) * alpha

    def sample(self):
        dirichlet = np.zeros(self.size, dtype=np.float64)

        # prange comes from numba and allows to parallelize the cycle
        for i in prange(self.size):
            # draw the gamma for each component
            dirichlet[i] = np.random.gamma(1, self.alpha[i])

        # normalize each component
        dirichlet = dirichlet / np.sum(dirichlet)

        # due to numerical instablity we set the first component as a residual in order
        # to make the vector sum to 1
        # dirichlet[0] += (1- np.sum(dirichlet))

        return dirichlet

    def update(self, indexes, step):
        # method used in order to make more likely a future draw from a given node
        # indexes should be a tuple conatining the indexes of the linked edges

        self.alpha[indexes[0]] += step
        self.alpha[indexes[1]] += step


############################ MULTINOMIAL ########################

spec2 = [
    ('size', int64),
    ('pdist', float64[:])
]


# We also define a multinomial class in order to draw the edges to link according to the prob distr
# sampled from the dirichlet
@jitclass(spec2)
class Multinomial(object):

    def __init__(self, size=None, pdist=None):
        
        if(pdist is None):
            self.pdist = np.ones(size) / size
        else:
            self.pdist = pdist


    def sample(self):
        # we draw from a multinomial and we extract the non 0 index, corresponding to the node to link
        draw1 = np.random.multinomial(n=1, pvals=self.pdist)
        edge_a = np.argwhere(draw1)[0][0] - 1

        draw2 = np.random.multinomial(n=1, pvals=self.pdist)
        edge_b = np.argwhere(draw2)[0][0] - 1

        return edge_a, edge_b

    def set_pvals(self, pdist):
        # used to set the prob dist drawn from the dirichlet
        self.pdist = pdist


############################ BINOMIAL ########################

spec3 = [('probs', float64[:])]


@jitclass(spec3)
class Binomial(object):

    def __init__(self, probs=np.ones(1)):
        self.probs = probs

    def sample(self):
        outcome = np.empty(len(self.probs), np.int8)

        for i, j in enumerate(self.probs):
            outcome[i] = np.random.binomial(n=1, p=j)

        return outcome
