from numba.experimental import jitclass
from numba import int64, float64
import numpy as np



#Non working functions that need to be revised




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




def timeseries_stats(self):
    lower_quant = []
    median = []
    upper_quant = []

    for i in range(50):
        prob_quantiles = np.percentile(self.infect_probs_over_time[i], [15, 25, 50, 75, 90])

        lower_quant.append(prob_quantiles[1])
        median.append(prob_quantiles[2])
        upper_quant.append(prob_quantiles[3])

    plt.plot(median, linewidth=2)  # mean curve.
    plt.fill_between(list(range(50)), lower_quant, upper_quant, color='b', alpha=.1)