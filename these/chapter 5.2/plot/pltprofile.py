import numpy as np
import matplotlib.pyplot as plt
import math

import os

import fitingfunction
import basefunction as bf




###########################################
#########  #########
###########################################
def plotprofile(dir, cf) :
	#################
	# plot
	fig = plt.figure(dir+'Line_concentration', figsize=[5.3, 7])
	fig.subplots_adjust(0.12, 0.08, 0.98, 1, .25, 0)

	### Detection Subfile of interest
	dirC = os.path.join( dir, 'concentration/' )
	os.chdir(dirC)
	(nFile, sFile) = bf.filestoanalyse( dirC, '*IMAGE*' )
	title = np.loadtxt('c_max.txt')
	number = np.loadtxt('video_c.txt', dtype=str)
	num = np.loadtxt('video_c.txt', dtype=int) - 1


	### Detection movie inside each file
	for n in nFile :
		# directory
		dirSub = os.path.join( dirC, n )
		os.chdir( dirSub )
		(nSubFile, sSubFile) = bf.filestoanalyse( dirSub, '*%*' )

		for nn in range(np.size(nSubFile)) :
		# sub directory
			nDirSub2 = os.path.join( dirSub, nSubFile[nn] )
			os.chdir( nDirSub2 )
			print( '\n \n -----------     -------------' )
			print( nDirSub2 )

		# constant value over the movie
			val = np.loadtxt('experimental_value.txt', skiprows=1)
			x = val[:,1]*1e-3
			tau = np.loadtxt('tau.txt')

		# load data
			# data
			for i in range(7) :
				bolus = np.loadtxt( 'bolusC_{}.txt'.format( number[i] ) )
				t = np.linspace( 0, np.size( bolus )/30, np.size( bolus ) )

				if i == 0 :
					maxBolus = np.max(bolus)
					cRatio =  title[nn,0] / maxBolus
				bolus = bolus * cRatio

				ptau = int(tau[int(number[i])]*30)
				ybot = np.zeros(ptau)
				ytop = bolus[:ptau]
				xplt = t[:ptau]

				k = i*3 + nn + 1
				ax = fig.add_subplot(7,3,k)
				ax.set_xlim( 0, title[nn,1] ) ; ax.set_ylim( 0, title[nn,0]+0.1*title[nn,0] )
				ax.fill_between(xplt, ybot, ytop, color=cf[nn], alpha=0.2)
				ax.plot(t, bolus, color=cf[nn])

				if k == 19 or k == 20 or k == 21 :
					ax.set_xlabel( r't (s)' )
				else :
					plt.setp( ax.get_xticklabels( ), visible=False )
					ax.xaxis.set_ticks_position( 'none' )

				if nn == 0 :
					ax.set_ylabel( r'$\Phi $' )
					ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))
				if nn == 1 :
					ax.yaxis.set_major_locator( plt.MultipleLocator( 0.5 ) )
				if nn == 2 :
					ax.yaxis.set_major_locator( plt.MultipleLocator( 2.0 ) )



	return fig

