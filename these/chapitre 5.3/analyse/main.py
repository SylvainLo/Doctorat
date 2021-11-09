import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing
import glob
import cv2
import math


import transformation_picture
import basefunction
import studymovie


####################################################################
### Global variable ###
	### number processor
num_cores = multiprocessing.cpu_count( )

	### plot latex
plt.rc( 'text', usetex=False )

	### figure
		# color
c = ['red', 'blue', 'green', 'magenta', 'gold', 'gray', 'cyan']

	### parameter study
backAnswer = basefunction.question('Do you want to create new backgrounds ? (1(yes)/0(no))')
channelAnswer = basefunction.question('Do you want to create a mask for differentiating the different channels ? (1(yes)/0(no))')
velocityAnswer = basefunction.question('Do you want to measure the velocity for each channel ? (1(yes)/0(no))')
intensityAnswer = basefunction.question('Do you want to measure the intensity for each channel ? (1(yes)/0(no))')
tauAnswer = basefunction.question('Do you want to measure a general tau ? (1(yes)/0(no))')
bolusAnswer = basefunction.question('Do you want to meausre a general bolus ? (1(yes)/0(no))')

	### variable experiment
sizePixCam = 5.86e-6
magnification = 4
pixToMicron = sizePixCam / magnification



####################################################################
### Main code ###
if __name__ == '__main__' :
	##################
	### Choice directory
	nameDirectory = 'G:/bolus_in_network/bolus_each_intersection_4mm/IMAGE_20-01-15/'              # nameDirectory = indicate the root of your data
	while os.path.isdir( nameDirectory ) is False :
		print( '   Error : your directory does not exist !' )
		nameDirectory = input( 'Directory of your data (E:/.../) : ' )
	os.chdir( nameDirectory )


	#################
	# Detection movie and data
	(nameMov, sMov) = basefunction.filestoanalyse(nameDirectory, '*video*')

	# data
	prop = np.loadtxt( 'experimental_value.txt', skiprows=1 ) 		     #[width entry (um), width channel(network in um), minimal length segment (um)]

	# data extrac from experiment
		# velocity
	if os.path.isfile( 'velocity.txt' ) :
		if velocityAnswer is '1' :
			measureV = True ; v = np.empty(sMov)
		else :
			measureV = False
			vReal = np.loadtxt('velocity.txt')
	else :
		measureV = True
		v = np.empty( sMov )

		# tau
	if os.path.isfile( 'tau.txt' ) :
		if tauAnswer is '1' :
			measureTau = True
			tau = np.empty(sMov) ; deltaTau = np.empty(sMov)
		else :
			measureTau = False
			tau = np.loadtxt('tau.txt') ; deltaTau = np.loadtxt('deltaTau.txt')
	else :
		measureTau = True
		tau = np.empty( sMov ) ; deltaTau = np.empty(sMov)


	### Study movie
	for n in range(sMov) :
		movie = cv2.VideoCapture( nameMov[n] )
		fps = int( movie.get( cv2.CAP_PROP_FPS ) )
		print( '	', nameMov[n], 'fps = ', fps )

	#################
	### Background
		if os.path.isfile('background_%02d.bmp'%n):
			if backAnswer is '1' :
				background = transformation_picture.backgroundcreation(movie)
				cv2.imwrite( 'background_%02d.bmp'%n, background )
			else :
				background = cv2.imread('background_%02d.bmp'%n, cv2.IMREAD_GRAYSCALE)
		else :
			background = transformation_picture.backgroundcreation(movie)
			cv2.imwrite( 'background_%02d.bmp'%n, background )


	#################
	### Part of interest
		if os.path.isfile( 'mask_channel_%02d.txt'%n ) and os.path.isfile('nbr_channel_analysis_%02d.txt'%n) :
			if channelAnswer is '1' :
				borneChannel, nbrChannel = studymovie.selectionpart(background, n)
			else :
				borneChannel = np.loadtxt( 'mask_channel_%02d.txt'%n, dtype=np.int )
				channelFile = open('nbr_channel_analysis_%02d.txt'%n, 'r')
				nbrChannel = int(channelFile.read())
				channelFile.close()
		else :
			borneChannel, nbrChannel = studymovie.selectionpart(background, n)


	#################
	### Velocity
		# Measure or search an existing solution for velocity
		if measureV is True :
			v[n] = studymovie.velocityanalysis(movie, n, borneChannel, prop)


	#################
	### Intensity variation
		# Intensity variation first movie
		if os.path.isfile( 'intensity_total_%02d.txt'%n) and os.path.isfile('concentration_totale_%02d.txt'%n) :
			if intensityAnswer is '1' :
				intensity, concentration = studymovie.intensityvariation(movie, background)
				np.savetxt('intensity_total_%02d.txt'%n, intensity)
				np.savetxt('concentration_totale_%02d.txt'%n, concentration)
			else :
				intensity = np.loadtxt('intensity_total_%02d.txt'%n)
				concentration = np.loadtxt('concentration_totale_%02d.txt'%n)
		else :
			intensity, concentration = studymovie.intensityvariation(movie, background)
			np.savetxt( 'intensity_total_%02d.txt'%n, intensity )
			np.savetxt( 'concentration_totale_%02d.txt'%n, concentration )


	#################
	### Unique bolus
		if os.path.isfile( 'bolus_I_%02d.txt'%n) and os.path.isfile('bolus_C_%02d.txt'%n) :
			if bolusAnswer is '1' :
				bolusI, bolusC = studymovie.functionbolus(intensity, concentration, prop)
				np.savetxt( 'bolus_I_%02d.txt'%n, intensity )
				np.savetxt( 'bolus_C_%02d.txt'%n, concentration )
			else :
				bolusI = np.loadtxt( 'bolus_I_%02d.txt'%n )
				bolusC = np.loadtxt( 'bolus_C_%02d.txt'%n )
		else :
			bolusI, bolusC = studymovie.functionbolus( intensity, concentration, prop )
			np.savetxt( 'bolus_I_%02d.txt'%n, intensity )
			np.savetxt( 'bolus_C_%02d.txt'%n, concentration )

	### size bolus
		# measure size bolus
		pdf = np.cumsum( bolusI ) ; pdf = pdf/pdf[-1]
		a = np.argmax( pdf > 0.80 )
		if measureTau is True :
			tau[n] = a/fps
			deltaTau[n] = tau[n]-tau[0]


	### save data
	if measureV is True :
		vReal = pixToMicron*prop[0:sMov,5]*v
		np.savetxt('velocity.txt', vReal, fmt='%.6f', delimiter='	')

	if measureTau is True :
		np.savetxt('tau.txt', tau, fmt='%.4f', delimiter='	')
		np.savetxt( 'deltaTau.txt', deltaTau, fmt='%.4f', delimiter='	' )

	plt.figure(1)
	plt.plot(prop[0:sMov,1], tau, 'o')
	plt.figure(2)
	plt.plot(prop[0:sMov,1], deltaTau, 'o')


	tauLine = np.loadtxt('tau_line.txt')
	xLine = np.loadtxt('experimental_value_line.txt', skiprows=1)
	plt.figure(3)
	plt.plot(prop[0:sMov,1], deltaTau*vReal, 'bo')
	plt.plot(xLine[:,1], (tauLine-tauLine[0])*4.04e-3, 'ro')

	plt.show()
