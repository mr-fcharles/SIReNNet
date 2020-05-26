import matplotlib.pyplot as plt
import numpy as np

def Binning( Z, N, d, extrema ):

	"""
		mu, c: vectors to keep track of data falling inside a bin
	           and the number of them into a bin
	           will be used to calculate the average of data inside the bin

		L:     number of observations in the temporal series

		extrmema: interval extrema, such that the i-th bin is [inex[i],inex[i+1]]
		       they exceed by 1 the number of bins
		       will be transformed in the new time for each observation
		       removing the last value

		delta: time step lenght, width of the bin

		loop1: here we collect all th eobservation into a bin summing their values into
		       the correspondent value of mu, and the number of observations into a bin
		       q is a switch ensuring that an observation hab been processed
		       for each observation will be false at the beginning, then when the time of an
		       observation has been allocated into a bin we do the sum and confirm the processing
		       setting q=True

		loop2: computes the average of observations that fall into a bin
		       here relies the weakness of the function: 
		       what happens if no observation fall into the bin?
		       the correspondent value will be zero, 
		       but one can introduce a third loop where the holes will be filled with the midpoint
		       or the line + some random noise

		       will throw away the first value of the output array, since we are obtaining
		       one new observation for each interval, Nint=Nbins-1

	"""

	X, t = Z[:,:-1], Z[:,-1]

	mu, c = np.zeros((N,d-1)), np.zeros(N)
	L = X.shape[0] 

	
	for i in range(L):

		q=False
		j=0
		while( q==False ):

			if(j>=N):
				break

			if(extrema[j] <= t[i] <= extrema[j+1]):

				mu[j]+= X[i,:]
				c[j] += 1
				q     = True

			else:
				j+=1

	for i in range(N): 

		if(c[i]!=0) : mu[i]/=c[i]
		else        : continue

	#if mu.any()==0:
		#continue
		#for() reconstruction loop

	return mu[1:]

def Repair(Z):
	"""
	se uno degli array ha dei buchi
	o dei salti troppo grossi riempi linearizzando 
	o interpolando
	"""

	N, d = Z.shape



	for i in range(N):

		if( Z[i,:].any() == 0 ):

			l, r = 0, 0

			while( Z[i-l,:].all() !=0): l+=1
			while( Z[i+r,:].all() !=0): r+=1

			Z[i,:] = 0.5*(Z[i-l,:]+Z[i+r,:])

	return Z

def GaussAugment( Z, d, T, delta, memory ):

	"""
	The functions augment the observations of a time series.
	Times are augmented with an exponential distribution, 
	according to the Gillespie algorithm, until they reach T

	Observations are sampled according a  multivariate "truncated" normal distribution:
	since we are doing the hypotesis that all nodes reached the stationary state
	so all observations that fall in the "memory time" of the functions are IID

	there is a truncation since only d-1 compartments are independent,
	the last is determined in order to keep total population conserved
	total population is calculated from the last observation 

	Mean is found averaging the last n=memory obs
	and variance according to the square root of the mean
	"""

	# VECCHIO SEI UN PAZZO NON ATTACCARE OGNI VOLTA L'EVOLUZIONE
	# PIUTTOSTO GENERI LA PARTE NUOVA A POI LA ATTACCHI ALLA FINE ALL'ARRAY

	pop = np.sum(Z[:-1,-1])
	X, t = Z[:,:-1], Z[:,-1]

	
	avg  = np.mean(X[:-memory,:],0)
	cov  = np.cov(X[:-memory,:],rowvar=False)

	while(t[-1]<T):
	# time update	
		
		#compartments update
		cmprt = list(range(d-1))
		g = np.random.choice( cmprt ) # scegli quale compartimento escludere 
		cmprt.remove(g)

		sigm = cov[cmprt,:]
		sigm = sigm[:,cmprt]
		mu   = avg[cmprt]

		y = np.zeros(d-1)
		y[cmprt] = np.random.multivariate_normal(mu,sigm) # multivariate gaussian for selected compartments
		y[g] = pop - np.sum(y[cmprt])
		y = y.reshape((1,d-1))

		X = np.concatenate((X,y))

		s = t[-1]
		s += np.random.uniform(0,delta) # cambia la regola in accordo con gillespie 
		t = np.append(t,s)
		

	t = t.reshape(t.shape[0],1)
	Z = np.concatenate((X,t),axis=1)

	return Z

def Augment( Z, d, T, delta, memory ):

	"""
	The functions augment the observations of a time series.
	Times are augmented with an exponential distribution, 
	according to the Gillespie algorithm, until they reach T

	One can hypotize that, if simulation reached the stationary state
	the time series of the single node can be augmented keeping the last state
	and sampling from a delta centered on this state with random time
	"""

	pop = np.sum(Z[:-1,-1])
	X, t = Z[:,:-1], Z[:,-1]

	
	avg  = np.mean(X[:-memory,:],0)
	cov  = np.cov(X[:-memory,:],rowvar=False)

	while(t[-1]<T):
	# time update	

		X = np.concatenate((X,X[-1:,:]))

		s = t[-1]
		s += np.random.uniform(0,delta) # cambia la regola in accordo con gillespie 
		t = np.append(t,s)
		

	t = t.reshape(t.shape[0],1)
	Z = np.concatenate((X,t),axis=1)

	return Z

# TIME SERIES PROCESSING, THE MAIN FUNCTION HERE
def TSPRCSS( F, tot_pop, T_SIMULATION): 

	""" 

	First part of the function is hjust some parameters setting.

	L:        number of samples of the longest time series
	TMAX,     TMIN: gratest and smalles lifespan of all the series
	Nbins:    number of bins in which the time series will be segmented
	memory:   parameter for augmentation. 
	            agmentation function will sample from a multivariate gaussian
	            with mean and covarianc ematrix calculated from the last #memory observations
	            of the time series.
	extrema:  bins are interval and will be represented by their extrema 	
	bin[i]:   [extrema[i],extrema[i+1]]

	Second part runs perform augmentation (if necessary) and binning of time series.
	The outcome is an np array with d=4 width and n=Nbins height 
	that represent the time series pf the whole system

	Third part is just a plot

	"""

	# PART 1
	communities = F.Nnodes()
	d           = F.dof()

	TMAX = max(F.all_local_times())
	TMIN = min(F.all_local_times())

	L = np.zeros(communities)

	for i in range(communities):
		L[i] = (F.trajectory(i)).shape[0]

	MAXL   = max(L)

    
	Nbins  = int( MAXL/50 )
	memory = 100

	delta  = (TMAX)/Nbins

	extrema = np.zeros(Nbins+1)
	

	for tau in range(Nbins):
		extrema[tau]=tau*delta

	extrema[Nbins]=TMAX

	print(" ========= AUGMENTATION & BINNING ========= ") 
	# PART 2

	P = np.zeros((Nbins-1,d))

	for m in range(communities):

		Q = np.zeros((d,Nbins))
		
		if( F.local_time(m) < TMAX):

			#Q = Augment( F.trajectory(m), d, TMAX, delta, memory )
			#Q = GaussAugment( F.trajectory(m), d, TMAX, delta, memory )
			Q = Binning( F.trajectory(m), Nbins, d, extrema )
			#plt.plot(range(Q.shape[0]),Q)
			#plt.show()
			P[:,:-1] += Q[:,:]/tot_pop  #numeric safety
			#continue
		else:

			Q = Binning( F.trajectory(m), Nbins, d, extrema )
			#plt.plot(extrema[:-2],Q)
			#plt.show
			P[:,:-1] += Q[:,:]/tot_pop  
		
	# PART3
	SIR_State = Repair(P)

	S    = SIR_State[:,0]
	I    = SIR_State[:,1]
	R    = SIR_State[:,2]
	time = extrema[:-2] #SIR_State[3,:]
	
	N = tot_pop
	
	plt.plot(time,S/N, label="S(t)")
	plt.plot(time,I/N, label="I(t)")
	plt.plot(time,R/N, label="R(t)")
	
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Percentage")
	plt.show()

	return
