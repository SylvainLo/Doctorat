import numpy as np

import math
import matplotlib.pyplot as plt




##################################
######### Displacement y #########
##################################
def vy(y, p):
	result = (p.mu_U/y**p.delta) * ( p.mu_A1 * np.sinh(p.b1 * (y-p.w_particle)) +
									 p.mu_A3 * np.sinh(p.b3 * (y-p.w_particle)) +
									 p.mu_A5 * np.sinh(p.b5 * (y-p.w_particle)) )
	return result