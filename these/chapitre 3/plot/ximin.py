import numpy as np

import matplotlib.pyplot as plt

import lmfit as lm

import functionfit



###############################
####### Variable global #######
###############################

### cte variable ###
N = 5
R0 = 2.8e-6





#######################
####### plot xi #######
#######################

### plot GridXi ###
def plotxi() :
	# value delta and xi
	xiRel = np.loadtxt( 'gridXiMean.txt' )
	gridDelta = np.loadtxt( 'gridDelta.txt' )

	### plot
	fig = plt.figure('mean xi relatif in function delta', figsize=[5, 4])
	fig.subplots_adjust( 0.16, 0.13, 0.97, 1, 0.15, 0.05 )
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$\delta$') ; ax.set_ylabel(r'$ \frac{ \sqrt{< (\xi - <\xi >)^2 >} }{<\xi >} $')
	ax.plot(gridDelta, xiRel, color='black', linestyle='', marker='o', markersize='5')

	return fig