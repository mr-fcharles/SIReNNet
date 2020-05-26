import numpy as np

"""
This function returns a matrix with the initial population for each node compartment
Tha population matrix is initialized at zero, and times columns is not changed anymore

a : adresses of non-infected cities

for loop1: initialize the non infected cities 
           sampling susceptible compartment from a uniform distribution

for loop2: initialize epicenter of epidemics
           with a mean percentage of infected people about 40%

"""

def RND_State0(epicenter, Communities, DOF, stats_parameters):
	
	MeanP = stats_parameters[0]
	DevP  = stats_parameters[1]

	epsilon = 10e-6
	# state
	S = np.zeros((Communities,DOF))

	# non ifected cities index
	a = np.delete( np.array(range(Communities)) , epicenter)

	
# susceptible cities population
	for i in a:

		population_size=np.random.randint(MeanP - DevP, MeanP + DevP)
		S[i,0] = population_size

# infected cities population
	for i in epicenter:

		population_size=np.random.randint(MeanP - DevP, MeanP + DevP)
		infected_percentage = 0.4 + 0.05*np.random.uniform(-1,1)

		S[i,0] = int((1-infected_percentage)*population_size)
		S[i,1] = population_size - S[i,0]


	return S
