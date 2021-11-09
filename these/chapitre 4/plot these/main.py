import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
import glob

import scipy.stats as sci

import functionfit as ff

import basefunction as bf

import plot as pl



####################################################################
### Global variable ###
	### number processor
num_cores = multiprocessing.cpu_count( )

	### plot latex
font = {'size':10}
plt.rc('font', **font)
rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(rc_params)


	### variable experiment
sizePixCam = 5.86 															   # size in um of a pixel of the camera
magnification = 32
pixToMicron = sizePixCam / magnification 										  # coefficient to transform pixel in um



####################################################################
### Main code ###
if __name__ == '__main__' :
	##################
	### plot 30 um
	fig30D, fig30C = pl.ploting('F:/dispersion/IMAGE_19-08-19_30um/', 10)

	os.chdir( 'F:/dispersion/' )
	fig30D.savefig( 'evolution_30.svg', format='svg' )
	fig30D.savefig( 'evolution_30.pdf', format='pdf' )
	fig30C.savefig( 'concentration_30.svg', format='svg' )
	fig30C.savefig( 'concentration_30.pdf', format='pdf' )


	##################
	### plot 20 um
	fig20D, fig20C = pl.ploting('F:/dispersion/IMAGE_19-08-20_20um/', 6.4)

	os.chdir( 'F:/dispersion/' )
	fig20D.savefig( 'evolution_20.svg', format='svg' )
	fig20D.savefig( 'evolution_20.pdf', format='pdf' )
	fig20C.savefig( 'concentration_20.svg', format='svg' )
	fig20C.savefig( 'concentration_20.pdf', format='pdf' )


	plt.show()