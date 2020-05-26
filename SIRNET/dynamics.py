import random
import numpy as np
# traiettoria random

class System:

	def __init__(self, z0 ):

		z = []
		self.N = z0.shape[0]
		for i in range( z0.shape[0] ):
			z.append((z0[i,:]))

		self.DOF = z0.shape[1]
		self.z0 = z0
		self.z = z

	# nel momento in cui è implementata quella a più nodi questa è inutile, appena finisci implementaz elimina
	def evolve_node( self, m, e ):
		self.z[m] = np.append(self.z[m],e)
		#self.z[m] = np.concatenate((self.z[m],e))
		return

	# lista con indici nodi da evolvere
	# array con i nuovi stati da appendere
	# occhio: tirare fuori omega e poi far evolvere tutto 
	# devono avere in ingresso la stessa lista dei nodi primi vinici
	# la lista la creo una volta sola nell'evoluzione

	def instant_state( self, m ):

		return self.z[m][-4:]

	def instant_neibgourhood(self, ngb):

		size = len(ngb)
		d = self.DOF

		p = np.zeros( (size,d) )
		
		i = 0
		for n in ngb:
			p[i,:] = self.z[n][-4:]
			i+=1

		return p
	
	def instant_full_state( self ): #

		d = self.DOF
		N = self.N

		p = np.zeros((N,d))

		for i in range(N):
			p[i,:] = self.z[i][-4:]

		return p

# devo ritornare l'ultimo valore di ciascun array!
	def all_local_times( self  ):
		
		d = self.DOF
		N = self.N

		p = np.zeros(N)

		for i in range(N):
			# l'ultimo tempo è sempre storato, per ogni nodo
			# come ultimo elemento dell'array
			p[i] = self.z[i][-1]

		return p

	def local_time( self, m ):
		
		d = self.DOF
		#tau = self.z[m][d*t:d*(t+1)][3]
		tau = self.z[m][-4:][3]

		return tau

	def trajectory( self, m ):

		l = int(len(self.z[m])/self.DOF)

		return np.array(self.z[m]).reshape(l,self.DOF)
	
	def initial_state(self):

		return self.z0

	def get_history(self):

		return self.z

	def dof(self):

		return self.DOF

	def Nnodes(self):

		return self.N
