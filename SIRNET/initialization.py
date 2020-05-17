import numpy as np


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
		S[i,3] = np.random.uniform(0,10e-6)

# infected cities population
	for i in epicenter:

		population_size=np.random.randint(MeanP - DevP, MeanP + DevP)
		infected_percentage = 0.4 + 0.05*np.random.uniform(-1,1)

		S[i,0] = int((1-infected_percentage)*population_size)
		S[i,1] = population_size - S[i,0]

		S[i,3] = np.random.uniform(0,10e-6)


	return S
