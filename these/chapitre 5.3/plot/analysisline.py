import numpy as np

import os

import matplotlib.pyplot as plt

import multiprocessing

import scipy.integrate as sci

import cv2

import math


import transformation_picture
import basefunction as bf
import studymovie
import fitingfunction



####################################################################
### Main code ###
def analysisline(file, fit, ax, color, marker, name ) :

	#################
	### analysis file 1
		# directory
	os.chdir( file )

		# detection movie and picture
	(nameMov, sMov) = bf.filestoanalyse(file, '*video*')

		# constant value over the movie
	val = np.loadtxt('experimental_value.txt', skiprows=1)
	xExp = val[:,1]*1e-6
	HExp = val[0,2]*1e-6 / 2
	R0 = 2.8e-6

		# velocity
	if os.path.isfile( 'velocity_pixel.txt' ) and os.path.isfile( 'velocity.txt' ) :
		vTic = True
		vPix = np.loadtxt('velocity_pixel.txt')
		v = np.loadtxt( 'velocity.txt' )
	else :
		vTic = False
		vPix = np.ones( sMov )

		# tau
	if os.path.isfile( 'tau.txt' ) and os.path.isfile( 'deltaTau.txt' ) and os.path.isfile( 'L.txt' ):
		tauTic = True ; tau = np.loadtxt( 'tau.txt' ) ; deltaTau = np.loadtxt( 'deltaTau.txt')
		if fit is True :
			v1mean = np.loadtxt('v1mean.txt') ; v0mean = np.loadtxt('v0mean.txt')
			L = np.loadtxt('L.txt')
	else :
		tauTic = False ; tau = np.ones( sMov ) ; deltaTau = np.ones( sMov )
		if fit is True :
			v1mean = np.ones( sMov )
			L = np.ones( sMov )


		# plot
	nPlot = int( math.ceil(sMov/4))

	figI = plt.figure('intensity line 30 um')
	figC = plt.figure('concentration line 30 um')
	figB = plt.figure('bolus line 30 um')
	figCum = plt.figure('cumsum bolus line 30 um')


	#################
	### Analysis of each movie
	for m in range(sMov) :

		# Opening of the movie
		movie = cv2.VideoCapture(nameMov[m])
		fps = int(movie.get(cv2.CAP_PROP_FPS))
		print('	',nameMov[m], 'fps = ', fps)


		# velocity
		if vTic is False :
			bf.exvelocity('FRONT')
			vPix[m] = studymovie.velocityanalysis( movie, m, val )


		# Background of the movie
		if os.path.isfile( 'background_%02d.bmp'%m ):
			background = cv2.imread('background_%02d.bmp'%m, cv2.IMREAD_GRAYSCALE)
		else :
			background = transformation_picture.backgroundcreation(movie)
			cv2.imwrite( 'background_%02d.bmp'%m, background )


		# Detection edges
		if os.path.isfile('wall-top_%02d.txt'%m) and os.path.isfile('wall-bot_%02d.txt'%m) :
			edgesTop = np.loadtxt('wall-top_%02d.txt'%m, dtype=np.int)
			edgesBot = np.loadtxt('wall-bot_%02d.txt'%m, dtype=np.int )
		else :
			edgesTop, edgesBot = transformation_picture.detectionedges( background )
			np.savetxt( 'wall-top_%02d.txt'%m, edgesTop, delimiter='	' )
			np.savetxt( 'wall-bot_%02d.txt'%m, edgesBot, delimiter='	' )


			# crop image to take center of the channel
		midChannel = int( np.mean (np.add(edgesBot, edgesTop)/2) )
		limTot = abs( int( np.mean(np.subtract(edgesBot, edgesTop))/2*0.95 ) )
		limTB = [midChannel-limTot, midChannel+limTot]


		# Intensity variation
		if os.path.isfile( 'intensity_%02d.txt'%m ) and os.path.isfile( 'concentration_%02d.txt'%m ):
			intensity = np.loadtxt('intensity_%02d.txt'%m)
			concentration = np.loadtxt('concentration_%02d.txt'%m)
		else :
			intensity, concentration = studymovie.intensityvariation(movie, background[limTB[0]:limTB[1],:], limTB)
			np.savetxt( 'intensity_%02d.txt'%m, intensity, fmt='%.4f', delimiter='	' )
			np.savetxt( 'concentration_%02d.txt'%m, concentration, fmt='%.4f', delimiter='	' )

		axI = figI.add_subplot(nPlot, 5, m+1)
		axI.plot( intensity)
		axC = figC.add_subplot(nPlot, 5, m+1)
		axC.plot( concentration )


		# Unique bolus
		axB = figB.add_subplot(nPlot, 5, m+1)
		axB2 = axB.twinx()
		axCum = figCum.add_subplot(nPlot, 5, m+1)

		if os.path.isfile( 'bolusI_%02d.txt'%m ) and os.path.isfile( 'cumsum_%02d.txt'%m ):
			bolusI = np.loadtxt( 'bolusI_%02d.txt'%m )
			bolusC = np.loadtxt( 'bolusC_%02d.txt'%m )
			pdf = np.loadtxt('cumsum_%02d.txt'%m)
		else :
			bolusI, bolusC = studymovie.functionbolus( intensity, concentration, val, axB )
			np.savetxt( 'bolusI_%02d.txt'%m, bolusI, fmt='%.4f', delimiter='	' )
			np.savetxt( 'bolusC_%02d.txt'%m, bolusC, fmt='%.4f', delimiter='	' )
			pdf = np.cumsum( bolusI ) ; pdf = pdf/pdf[-1]
			np.savetxt( 'cumsum_%02d.txt'%m, pdf, fmt='%.4f', delimiter='	' )


		axB.plot(bolusI,'black', linewidth=2)
		axB2.plot(bolusC, 'red', linewidth=2)
		axCum.plot(pdf)

		# measure size bolus
		a = np.argmax( pdf > 0.80 )
		if tauTic is False :
			tau[m] = a / fps
			deltaTau[m] = tau[m] - tau[0]

		print('mean concentration pulse', np.mean(bolusC[0:a]))
		print('value concentration at 80%', bolusC[a])


	### Save result
		# velocity
		if vTic is False :
			v = vPix*val[:, 5]*1e-6/val[:, 4]*fps
			np.savetxt( 'velocity.txt', v ) ; np.savetxt('velocity_pixel.txt', vPix)

		# value for plot and fit
		if tauTic is False :
			# tau, deltaTau
			np.savetxt('tau.txt', tau) ; np.savetxt('deltaTau.txt', deltaTau)

			if fit is True :
			# v0mean(x)
				v0mean=sci.cumtrapz(v,xExp,initial=v[0]) ; v0mean[1:]=v0mean[1:]/xExp[1:] ; np.savetxt('v0mean.txt', v0mean)

			# v1mean(x)
				bf.exvelocity('back')
				v1mean[1:] = ( xExp[1:] * v0mean[1:] ) /  ( deltaTau[1:] * v0mean[1:] + xExp[1:])
				v1mean[0] = studymovie.velocityanalysis( cv2.VideoCapture(nameMov[0]), 0, val ) * \
						val[0, 5]*1e-6/val[0, 4]*fps ; np.savetxt('v1mean.txt', v1mean)

			# L(x)
				L[1:] = ( ((3*HExp**2-R0**2)* xExp[1:]) / 6 ) * v1mean[1:] ; L[0] = 0 ; np.savetxt('L.txt', L)



	### tot result
		ax.plot( xExp, deltaTau, label='{}'.format(name), marker=marker, color=color )


	### Fitting result
		if fit is True :
		# z initial
			z0 = HExp - math.sqrt( HExp**2 * (1-v1mean[0]/v[0]) - R0**2/3 )

		# result to fit
			result = fitingfunction.fitingresult(L, val, z0, v)

		# velocity
			v1meantheo = fitingfunction.zfit2(xExp, result, v)


		# deltaTau
			deltaTauThe = xExp * (1/v1meantheo - 1/v0mean)
			ax.plot( xExp, deltaTauThe, linestyle=':', label='fit {}'.format(name), color=color )


	ax.legend( )

	return ax