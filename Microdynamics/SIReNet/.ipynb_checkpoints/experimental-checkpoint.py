from numba.experimental import jitclass
from numba import int64, float64
from numba import prange
import numpy as np


spec4 = [
    ('size', int64),
    ('pdist', float64[:]),
    ('randomization', float64)
]

@jitclass(spec4)
class Multinomial_with_randomization(object):

    def __init__(self, size=1, pdist= None,randomization=0):
        
        if(pdist is None):
            self.pdist = np.ones(size) / size
            
        else:
            self.pdist = pdist
            
        self.randomization  = randomization

    def sample(self):

        randomization_indicator = np.random.binomial(p=self.randomization,n=1)

        if(randomization_indicator==0):

            # we draw from a multinomial and we extract the non 0 index, corresponding to the node to link
            draw1 = np.random.multinomial(n=self.size, pvals=self.pdist)
            edge_a = np.argwhere(draw1)[0][0]

            draw2 = np.random.multinomial(n=self.size, pvals=self.pdist)
            edge_b = np.argwhere(draw2)[0][0]

            return edge_a, edge_b

        else:
            
            print('here')

            edge_a,edge_b = np.random.randint(low=0,high=self.size,size=2)

            return edge_a, edge_b

    def set_pvals(self, pdist):
        # used to set the prob dist drawn from the dirichlet
        self.pdist = pdist

        
spec2 = [
    ('size', int64),
    ('pdist', float64[:])
]


# We also define a multinomial class in order to draw the edges to link according to the prob distr
# sampled from the dirichlet
@jitclass(spec2)
class Multinomial(object):

    def __init__(self, size=None, pdist=None):
        self.pdist = np.ones(size) / size
        self.size = size

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