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
### Global variable ###
	### number processor
num_cores = multiprocessing.cpu_count( )

	### plot latex
plt.rc( 'text', usetex=False )

	### figure
		# color
c = ['darkblue', 'blue', 'cyan', 'darkred', 'red', 'darkgreen', 'limegreen', 'lime']

	### parameter study
velocityAnswer = bf.question('Do you want to measure the v0(x) ? (1(yes)/0(no))')
tauAnswer = bf.question('Do you want to measure tau(x) ? (1(yes)/0(no))')
backAnswer = bf.question('Do you want to create new backgrounds ? (1(yes)/0(no))')
wallAnswer = bf.question('Do you want to detect the position of the wall ? (1(yes)/0(no))')
intensityAnswer = bf.question('Do you want to measure the I/I0 (intensity) ? (1(yes)/0(no))')
bolusAnswer = bf.question('Do you want to reconstruct a mean bolus ? (1(yes)/0(no))')

	### plot
fig1 = plt.figure('tau')
ax1 = fig1.add_subplot(111)

fig2 = plt.figure('velocity min')
ax2 = fig2.add_subplot(111)

fig3 = plt.figure('deltaTau')
ax3 = fig3.add_subplot(111)

fig4 = plt.figure('velocity max')
ax4 = fig4.add_subplot(111)

fig5 = plt.figure('L')
ax5 = fig5.add_subplot(111)

fig6 = plt.figure('rapport velocity')
ax6 = fig6.add_subplot(111)




####################################################################
### Main code ###
if __name__ == '__main__' :
	##################
	### Choice directory
	nameDir = 'F:/bolus_concentration/concentration'
	while os.path.isdir( nameDir ) is False :
		print( '   Error : your directory does not exist !' )
		nameDir = input( 'Directory of your data (E:/.../) : ' )
	os.chdir( nameDir )


	#################
	### Detection Subfile of interest
	(nameFile, sFile) = bf.filestoanalyse( nameDir, '*IMAGE*' )

	cn = int( 0 )

	### Detection movie inside each file
	for n in nameFile :
		# directory
		nameDirSub = os.path.join( nameDir, n )
		os.chdir( nameDirSub )
		(nameSubFile, sSubFile) = bf.filestoanalyse( nameDirSub, '*%*' )

		for nn in nameSubFile :
		# sub directory
			nameDirSub2 = os.path.join( nameDirSub, nn )
			os.chdir( nameDirSub2 )
			print( nameDirSub2 )

		# detection movie and picture
			(nameMov, sMov) = bf.filestoanalyse(nameDirSub2, '*video*')

		# constant value over the movie
			val = np.loadtxt('experimental_value.txt', skiprows=1)
			xExp = val[:,1]*1e-6
			HExp = val[0,2]*1e-6 / 2
			R0 = 2.8e-6

		# velocity
			if os.path.isfile( 'velocity_pixel.txt' ) and os.path.isfile( 'velocity.txt' ) :
				if velocityAnswer is '1' :
					vTic = False
					vPix = np.ones( sMov )
				else :
					vTic = True
					vPix = np.loadtxt('velocity_pixel.txt')
					v = np.loadtxt( 'velocity.txt' )
			else :
				vTic = False
				vPix = np.ones( sMov )

		# tau
			if os.path.isfile( 'tau.txt' ) and os.path.isfile( 'deltaTau.txt' ) and os.path.isfile( 'L.txt' ):
				if tauAnswer is '1' :
					tauTic = False
					tau = np.ones( sMov )
					deltaTau = np.ones( sMov )
					v1mean = np.ones( sMov)
					L = np.ones( sMov)
				else :
					tauTic = True
					tau = np.loadtxt( 'tau.txt' )
					deltaTau = np.loadtxt( 'deltaTau.txt')
					v1mean = np.loadtxt('v1mean.txt')
					L = np.loadtxt('L.txt')
					v0mean = np.loadtxt('v0mean.txt')
			else :
				tauTic = False
				tau = np.ones( sMov )
				deltaTau = np.ones( sMov )
				v1mean = np.ones( sMov )
				L = np.ones( sMov )


		# plot
			nPlot = int( math.ceil(sMov/5))

			figI = plt.figure('intensity' + n + nn )
			figC = plt.figure('concentration' + n + nn )
			figB = plt.figure('bolus' + n + nn )
			figCum = plt.figure('cumsum bolus' + n + nn )



	### Analysis of each movie
			for m in range(sMov) :

		# Opening of the movie
				movie = cv2.VideoCapture(nameMov[m])
				fps = int(movie.get(cv2.CAP_PROP_FPS))
				print('	',nameMov[m], 'fps = ', fps)


		# velocity
				if vTic is False :
					bf.exvelocity('front')
					vPix[m] = studymovie.velocityanalysis( movie, m, val )


		# Background of the movie
				if os.path.isfile( 'background_%02d.bmp'%m ):
					if backAnswer is '1' :
						background = transformation_picture.backgroundcreation(movie)
						cv2.imwrite( 'background_%02d.bmp'%m, background )
					else :
						background = cv2.imread('background_%02d.bmp'%m, cv2.IMREAD_GRAYSCALE)
				else :
					background = transformation_picture.backgroundcreation(movie)
					cv2.imwrite( 'background_%02d.bmp'%m, background )


		# Detection edges
			# detection edges
				if os.path.isfile('wall-top_%02d.txt'%m) and os.path.isfile('wall-bot_%02d.txt'%m) :
					if wallAnswer is '1' :
						edgesTop, edgesBot = transformation_picture.detectionedges( background )
						np.savetxt('wall-top_%02d.txt'%m, edgesTop, delimiter='	')
						np.savetxt('wall-bot_%02d.txt'%m, edgesBot, delimiter='	')
					else :
						edgesTop = np.loadtxt('wall-top_%02d.txt'%m, dtype=np.int)
						edgesBot = np.loadtxt('wall-bot_%02d.txt'%m, dtype=np.int )
				else :
					edgesTop, edgesBot = transformation_picture.detectionedges( background )
					np.savetxt( 'wall-top_%02d.txt'%m, edgesTop, delimiter='	' )
					np.savetxt( 'wall-bot_%02d.txt'%m, edgesBot, delimiter='	' )


			# crop image to take center of the channel
				midChannel = int( np.mean (np.add(edgesBot, edgesTop)/2) )
				limTot = abs( int( np.mean(np.subtract(edgesBot, edgesTop))/2*0.5 ) )

				limTB = [midChannel-limTot, midChannel+limTot]


		# Intensity variation
				if os.path.isfile( 'intensity_%02d.txt'%m ) and os.path.isfile( 'concentration_%02d.txt'%m ):
					if intensityAnswer is '1' :
						limX = int(vPix[m]*1.2)
						intensity, concentration = studymovie.intensityvariation(movie, background[limTB[0]:limTB[1],:], limTB)
						np.savetxt( 'intensity_%02d.txt'%m, intensity, fmt='%.4f', delimiter='	' )
						np.savetxt( 'concentration_%02d.txt'%m, concentration, fmt='%.4f', delimiter='	' )
					else :
						intensity = np.loadtxt('intensity_%02d.txt'%m)
						concentration = np.loadtxt('concentration_%02d.txt'%m)
				else :
					intensity, concentration = studymovie.intensityvariation(movie, background[limTB[0]:limTB[1],:], limTB)
					np.savetxt( 'intensity_%02d.txt'%m, intensity, fmt='%.4f', delimiter='	' )
					np.savetxt( 'concentration_%02d.txt'%m, concentration, fmt='%.4f', delimiter='	' )

				# axI = figI.add_subplot(nPlot, 5, m+1)
				# axI.plot( intensity)
				axC = figC.add_subplot(nPlot, 5, m+1)
				axC.plot( concentration )


		# Unique bolus
				axB = figB.add_subplot(nPlot, 5, m+1)
				axB2 = axB.twinx()
				axCum = figCum.add_subplot(nPlot, 5, m+1)

				if os.path.isfile( 'bolusI_%02d.txt'%m ) and os.path.isfile( 'cumsum_%02d.txt'%m ):
					if bolusAnswer is '1' :
						bolusI, bolusC = studymovie.functionbolus( intensity, concentration, val, axB)
						np.savetxt( 'bolusI_%02d.txt'%m, bolusI, fmt='%.4f', delimiter='	' )
						np.savetxt( 'bolusC_%02d.txt'%m, bolusC, fmt='%.4f', delimiter='	' )
						pdf = np.cumsum( bolusI ) ; pdf = pdf/pdf[-1]
						np.savetxt( 'cumsum_%02d.txt'%m, pdf, fmt='%.4f', delimiter='	' )
					else :
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
				np.savetxt( 'velocity.txt', v )
				np.savetxt('velocity_pixel.txt', vPix)

		# value for plot and fit
			if tauTic is False :
			# tau, deltaTau
				np.savetxt('tau.txt', tau)
				np.savetxt('deltaTau.txt', deltaTau)

			# v0mean(x)
				v0mean = sci.cumtrapz( v, xExp, initial=v[0] )
				v0mean[1:] = v0mean[1:]/xExp[1:]
				np.savetxt('v0mean.txt', v0mean)

			# v1mean(x)
				bf.exvelocity('back')
				v1mean[1:] = ( xExp[1:] * v0mean[1:] ) /  ( deltaTau[1:] * v0mean[1:] + xExp[1:])
				v1mean[0] = studymovie.velocityanalysis( cv2.VideoCapture(nameMov[0]), 0, val ) * val[0, 5]*1e-6/val[0, 4]*fps
				np.savetxt('v1mean.txt', v1mean)

			# L(x)
				L[1:] = ( ((3*HExp**2-R0**2)* xExp[1:]) / 6 ) * v1mean[1:]
				L[0] = 0
				np.savetxt('L.txt', L)



	### tot result
			ax1.plot( xExp, tau, label=n[-4:]+nn, color=c[cn] )
			ax2.plot( xExp, v1mean, label=n[-4:]+nn, color=c[cn] )
			ax3.plot( xExp, deltaTau, label=n[-4:]+nn, color=c[cn] )
			ax4.plot( xExp, v, label=n[-4:]+nn, color=c[cn] )
			ax4.plot( xExp, v0mean, linestyle=':' ,label=n[-4:]+nn, color=c[cn])
			ax5.plot( xExp, L, label=n[-4:]+nn, color=c[cn] )
			ax6.plot( xExp, v1mean/v0mean, label=n[-4:]+nn, color=c[cn])


	### Fitting result
		# z initial
			z0 = HExp - math.sqrt( HExp**2 * (1-v1mean[0]/v[0]) - R0**2/3 )

		# result to fit
			result = fitingfunction.fitingresult(L, val, z0, v)
			LNum = fitingfunction.zfit(xExp, result, v)
			ax5.plot( xExp, LNum, linestyle=':', label=n[-4 :]+nn, color=c[cn] )

		# velocity
			v1meantheo = fitingfunction.zfit2(xExp, result, v)
			ax2.plot( xExp, v1meantheo, linestyle=':', label=n[-4:]+nn, color=c[cn] )

		# rapport velocity
			rV = v1meantheo/v0mean
			ax6.plot( xExp, rV, linestyle=':', label=n[-4:]+nn, color=c[cn] )

		# deltaTau
			deltaTauThe = xExp * (1/v1meantheo - 1/v0mean)
			ax3.plot( xExp, deltaTauThe, linestyle=':', label=n[-4 :]+nn, color=c[cn] )

	### constnt color
			cn += int(1)

	ax1.legend( )
	ax2.legend( )
	ax3.legend( )
	ax4.legend( )
	ax5.legend( )
	ax6.legend( )
	plt.show()