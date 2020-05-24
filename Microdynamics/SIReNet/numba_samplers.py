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
    '''
    Costumly defined class used to draw probability distributions over a simplex
    Compiled with numba (experimental)


    Attributes
    ___________

    - size: int
        Dimension of the simplex

    - alpha: float > 0
        Parameter of the Dirichlet distribution.

    Methods
    __________

    - sample():
        Samples a distribution (array summing up to one) from the Dirichlet distr

    - update(indexes: tuple,list,array - step: float > 0):
        Updates alpha in order to achieve preferential attachment

    '''

    def __init__(self, size=10, alpha=0.1):
        self.size = size
        self.alpha = np.ones(self.size) * alpha

    def sample(self):
        '''
        Samples a distribution (array summing up to one) from the Dirichlet distr

        :return: np.array
        '''

        dirichlet = np.zeros(self.size, dtype=np.float64)

        # prange comes from numba and allows to parallelize the cycle
        for i in prange(self.size):
            # draw the gamma for each component
            dirichlet[i] = np.random.gamma(1, self.alpha[i])

        # normalize each component (we obtain a dirichlet by transforming gammas)
        dirichlet = dirichlet / np.sum(dirichlet)

        # due to numerical instablity we add the first component the normalization residual in order
        # to make the vector sum to 1
        dirichlet[0] += (1- np.sum(dirichlet))

        return dirichlet

    def update(self, indexes, step):
        '''
        Updates alpha in order to achieve preferential attachment

        :param indexes: tuple, list or array
        :param step: float > 0
        '''
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
    '''
    Costumly defined classed used to draw edges extremity from a specified distribution

    Attributes
    ___________

    - size: int
        Deprecated # todo fix use of size in Multinomial class

    - pdist: np.array
        Probabiity distribution over a finite number of elements

    Methods
    __________

    - sample():
        Samples the pair of nodes to link with an edge

    - set_pvals(pdist: np.array):
        Updates the probability distribution from which the sampling is performed


    '''

    def __init__(self, size=None, pdist=None):

        if (pdist is None):
            self.pdist = np.ones(size) / size
        else:
            self.pdist = pdist

    def sample(self):
        '''
        Samples the pair of nodes to link with an edge

        :return: tuple
        '''

        # we draw from a multinomial and we extract the only non 0 index, corresponding to the node id
        draw1 = np.random.multinomial(n=1, pvals=self.pdist)
        #multinomial draws between 1 and len(self.pdist), our indexes start from 0 and go up to len(self.pdist)-1
        edge_a = np.argwhere(draw1)[0][0] - 1

        draw2 = np.random.multinomial(n=1, pvals=self.pdist)
        edge_b = np.argwhere(draw2)[0][0] - 1

        return edge_a, edge_b

    def set_pvals(self, pdist):
        '''
        Updates the probability distribution from which the sampling is performed

        :param pdist: np.array
        '''
        # used to set the prob dist drawn from the dirichlet
        self.pdist = pdist



############################ BINOMIAL ########################

spec3 = [('probs', float64[:])]


@jitclass(spec3)
class Binomial(object):
    '''
    Costumly define class used to draw multiple Bernoulli draws each wrt to its own prob

    Attributes
    ___________

    - probs: np.array
        Arrays containing the prob of 1 for each of the bernoulli flips

    Methods
    __________

    - sample():
        Returns an array of 0-1 values for each of the specified probs

    '''

    def __init__(self, probs=np.ones(1)):
        self.probs = probs

    def sample(self):

        '''
        Returns an array of 0-1 values for each of the specified probs

        :return: np.array
        '''

        outcome = np.empty(len(self.probs), np.int8)

        for i, j in enumerate(self.probs):
            outcome[i] = np.random.binomial(n=1, p=j)

        return outcome
