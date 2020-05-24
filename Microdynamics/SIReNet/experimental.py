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
        time_series_array = np.sin(np.linspace(-np.pi, np.pi, 400)) + np.random.rand((400))
        n_steps = 15  # number of rolling steps for the mean/std.

        # Compute curves of interest:
        time_series_df = pd.DataFrame(time_series_array)
        smooth_path = time_series_df.rolling(n_steps).mean()
        path_deviation = 2 * time_series_df.rolling(n_steps).std()

        under_line = (smooth_path - path_deviation)[0]
        over_line = (smooth_path + path_deviation)[0]

        # Plotting:
        plt.plot(smooth_path, linewidth=2)  # mean curve.
        plt.fill_between(path_deviation.index, under_line, over_line, color='b', alpha=.1)

 