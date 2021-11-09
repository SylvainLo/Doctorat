import numpy as np

import os

import matplotlib.pyplot as plt

import multiprocessing

import scipy.integrate as sci

import cv2

import math


import fitingfunction
import basefunction as bf




def test(dir, col, mark) :
	#################
	# plot
	fig = plt.figure( dir+'Line_concentration', figsize=[5.3, 7] )
	fig.subplots_adjust( 0.12, 0.08, 0.98, 1, .25, 0 )

	### Detection Subfile of interest
	dirT = os.path.join( dir, 'density/IMAGE_19-10-27/' )
	os.chdir( dirT )
	(nSubFile, sSubFile) = bf.filestoanalyse( dirT, '*%*' )

	### value of interest for xi
	xiVal = [35,40,45,50,55,60,65,70,75,80,85,90,95]

	for nn in range(np.size(nSubFile)) :
		# sub directory
		nDirSub2 = os.path.join( dirT, nSubFile[nn] )
		os.chdir( nDirSub2 )
		print( nDirSub2 )

		# constant value over the movie
		val = np.loadtxt('experimental_value.txt', skiprows=1)
		xExp = val[:,1]*1e-6
		HExp = val[0, 2]*1e-6/2
		R0 = 2.8e-6

		v = np.loadtxt('velocity.txt')
		v0mean = np.loadtxt( 'v0mean.txt' )
		v1mean0 = np.loadtxt( 'v1mean.txt' )

		z0 = HExp-math.sqrt( HExp**2*(1-v1mean0[0]/v0mean[0])-R0**2/3 )

		(nFileBol, sFileBol) = bf.filestoanalyse( nDirSub2, '*cumsum*' )
		# load data
			# data
		for i in xiVal :
			deltaTau = np.empty(sFileBol)

			for n in range(sFileBol) :
				bolus = np.loadtxt( nFileBol[n] )*100

				a = np.argmax( bolus > i )
				if n == 0 :
					tau0 = a/30

				tau = a/30
				deltaTau[n] = tau-tau0

				# velocity head and tail bolus
			v1mean = np.ones( sFileBol ) ; v1mean[0] = v1mean0[0]
			v1mean[1 :] = (xExp[1 :]*v0mean[1 :])/(deltaTau[1 :]*v0mean[1 :]+xExp[1 :])

				# L(x)
			L = np.ones( sFileBol ) ; L[1 :] = (((3*HExp**2-R0**2)*xExp[1 :])/6)*v1mean[1 :] ; L[0] = 0

				# fit
			print( 'result for ', i, ' is ')
			result = fitingfunction.fitingresult( L, val, z0, v )


				# plot
			plt.plot(i, result['xi'].value, marker=mark[nn], color=col[nn])

	return fig




def onlyone(dir, col, mark) :
	#################
	# plot
	fig = plt.figure( 'test', figsize=[5.3, 7] )
	fig.subplots_adjust( 0.12, 0.08, 0.98, 1, .25, 0 )

	### Detection Subfile of interest
	dirT = os.path.join( dir, 'density/IMAGE_19-10-27/' )
	os.chdir( dirT )
	(nSubFile, sSubFile) = bf.filestoanalyse( dirT, '*%*' )

	### value of interest for xi
	xiVal = [35,40,45,50,55,60,65,70,75,80,85,90,95]


		# sub directory
	nDirSub2 = os.path.join( dirT, nSubFile[0] )
	os.chdir( nDirSub2 )
	print( nDirSub2 )

		# constant value over the movie
	val = np.loadtxt('experimental_value.txt', skiprows=1)
	xExp = val[:,1]*1e-6
	HExp = val[0, 2]*1e-6/2
	R0 = 2.8e-6

	v = np.loadtxt('velocity.txt')
	v0mean = np.loadtxt( 'v0mean.txt' )
	v1mean0 = np.loadtxt( 'v1mean.txt' )

	z0 = HExp-math.sqrt( HExp**2*(1-v1mean0[0]/v0mean[0])-R0**2/3 )

	(nFileBol, sFileBol) = bf.filestoanalyse( nDirSub2, '*cumsum*' )
		# load data
			# data
	for i in xiVal :
		deltaTau = np.empty(sFileBol)

		for n in range(sFileBol) :
			bolus = np.loadtxt( nFileBol[n] )*100

			a = np.argmax( bolus > i )
			if n == 0 :
				tau0 = a/30

			tau = a/30
			deltaTau[n] = tau-tau0

				# velocity head and tail bolus
		v1mean = np.ones( sFileBol ) ; v1mean[0] = v1mean0[0]
		v1mean[1 :] = (xExp[1 :]*v0mean[1 :])/(deltaTau[1 :]*v0mean[1 :]+xExp[1 :])

				# L(x)
		L = np.ones( sFileBol ) ; L[1 :] = (((3*HExp**2-R0**2)*xExp[1 :])/6)*v1mean[1 :] ; L[0] = 0

				# fit
		print( 'result for ', i, ' is ' )
		result = fitingfunction.fitingresult( L, val, z0, v )


				# plot
		plt.plot(i, result['xi'].value, marker=mark[0], color=col[0])




	return fig
