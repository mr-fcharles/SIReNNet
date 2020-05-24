#%%
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.filterwarnings('ignore',category=NumbaDeprecationWarning)

#%%
from SIReNet.epidemics_graph import EpdimecisGraph

prova = EpdimecisGraph(pop_size=1000)
prova.sampler_initializer()


#%%

prova.create_families()

#%%

prova.adjacency

#%%

prova.compute_common_neighbors()

#%%

prova.add_links(get_richer_step=0.01)

#%%
prova.build_nx_graph()
prova.degree_distribution()

#%%

prova.compute_common_neighbors()

#%%

prova.start_infection(contagion_probability=0.05)

#%%

prova.initialize_individual_factors()

#%%

prova.infect_over_time

#%%

prova.propagate_infection(mu=2)
prova.propagation_stats()



def funzione_prova():

    '''
    solo per provare i docs
    :return:
    '''

funzione_prova()