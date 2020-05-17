import random
import numpy as np
import networkx as nx
import math
#import scipy
import matplotlib.pyplot as plt

from gilevo import Gillespie
from initialization import RND_State0
from dynamics import System

from mat import random_network
from mat import neighbourhood_weights
from mat import first_neighbours



dof = 4
ngb = 6

a = np.random.randint(0,10,(dof,ngb))
b = np.random.randint(0,10,dof)

print(a)
print(np.sum(a,1))
c=np.sum(a,1)
print("iiiii",b+c)

