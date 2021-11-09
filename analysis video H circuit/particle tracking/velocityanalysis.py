import pandas ;
import numpy as np ;

import matplotlib.pyplot as plt ;

import os ;
import fonctionbase ;

import math ;






#####################################
######### Velocity Analysis #########
#####################################

	######### Velocity analysis #########

def velocityanalysis(nameFile, numeroOfFile, variableFile) :
	# Loading data
	dataTrack = pandas.read_csv(nameFile[numeroOfFile])	;
	print(nameFile[numeroOfFile]) ;

	distance = variableFile.distance

	# Introduction variable
	nbrTrack = dataTrack.numeroTrack.iat[-1] ; cte=0 ;
	velocityForTrack = np.zeros((int(nbrTrack), 4)) ; 

	# Creation vector for one track
	for iv in range(int(nbrTrack)) :
		dataForVelocityAnalysis = dataTrack[dataTrack.numeroTrack == iv]
		(rowTest,columnTest) = np.shape(dataForVelocityAnalysis) ;

		# Mean of the tracking
		if rowTest > 2 :
			velocityForTrack[cte,0] = iv+1 ; #numero track
			velocityForTrack[cte,1] = np.sum(dataForVelocityAnalysis['frame'])/rowTest ; #mean frame
			velocityForTrack[cte,2] = np.sum(dataForVelocityAnalysis['x'])/rowTest
			velocityForTrack[cte,3] = ((dataForVelocityAnalysis.x.iat[-1] - dataForVelocityAnalysis.x.iat[0]) / 
						(dataForVelocityAnalysis.frame.iat[-1] - dataForVelocityAnalysis.frame.iat[0])) ;
			cte += 1

			# Correction frame numero
	frameCorrection = ( np.mean(velocityForTrack[:,2]) - velocityForTrack[:,2] ) / velocityForTrack[:,3]
	velocityForTrack[:,1] = velocityForTrack[:,1] + frameCorrection		
		
			# Reduction size vector velocityForTrack (take only the one which are differnet form 0)
	condition = (velocityForTrack[:,3] != 0) ; 
	velocityField = np.compress(condition, velocityForTrack, axis=0)
	print('shape velocityField = ', np.shape(velocityField))
	
			# Sorting results
	sortingArray = np.argsort(velocityField[:,1]) ;
	velocityFieldSorted = np.array(velocityField[sortingArray]) ;

	# Save local data
	velocityLocalFile = 'velocityLocalFile_%03d.txt'%(numeroOfFile+1) 
	np.savetxt(velocityLocalFile, velocityFieldSorted, fmt='%.2f', delimiter='    ')

	return()







	######### Plot velocity #########

def plotvelocity(subdirectory, variableFile) :
	### Plot local variable ###
	os.chdir(subdirectory) ;

	# Searching data
	(nameDataToPlot, sizeDataToPlot) = fonctionbase.filestoanalyse(subdirectory, '*velocityLocalFile*') ;

	# Loading data
	distance = variableFile.distance
	line = len(variableFile.limit_velocity) ;

	# Creating global Data
	velocity = np.empty(sizeDataToPlot) ;
	velocityMean = np.empty(sizeDataToPlot) ;
	velocityStandarDeviation = np.empty(sizeDataToPlot) ;
	velocityForY = np.empty((sizeDataToPlot, line)) ;

	# Prepare plot
	nbrRow = math.ceil(sizeDataToPlot/3) ;

	# Load local data
	for j in range(sizeDataToPlot) :
		localData = np.loadtxt(nameDataToPlot[j]) ;
			
		# Plot
		plt.figure(1 + variableFile.n_files) ;
		plt.subplot(nbrRow, 3, j+1) ; 
		plt.suptitle('Velocity (pix/frame) depending the numero of the frame') ;
		plt.plot(localData[:,1],localData[:,3], 'o', color = variableFile.color_file[variableFile.n_files]) ; 

		# Plot
		plt.figure(10 + variableFile.n_files) ;
		plt.subplot(nbrRow, 3, j+1) ; 
		plt.suptitle('Velocity (pix/frame) depending the number of the track') ;
		plt.plot(localData[:,0],localData[:,3], 'o', color = variableFile.color_file[variableFile.n_files]) ; 

		# Plot histogram
			# Histogram
		(histV, binV) = np.histogram(abs(localData[:,3]), bins=variableFile.histogramm_bin, range=(0,350)) ;
		
			# Cumsum
		sumV = np.cumsum(histV)
		sumVNormalized = sumV/np.max(sumV)

		plt.figure(15 + variableFile.n_files) ;
		plt.suptitle('Velocity distribution') ;
		plt.plot(np.linspace(0,350,variableFile.histogramm_bin),sumVNormalized, color=variableFile.color_file_video[j])



		### Plot Global Data ###
			# Complete the global data
		velocity[j] = np.percentile(abs(localData[:,3]), 99) ;
		velocityMean[j] = np.mean(abs(localData[:,3])) ;
		velocityStandarDeviation[j] = np.std(abs(localData[:,3])) ;

		for i in range(line) :
			velocityForY[j,i] = np.percentile(abs(localData[:,3]), variableFile.limit_velocity[i]) ;		

			# Saving data
	np.savetxt('maximalVelocity.txt', velocity, fmt='%1.5f',delimiter='    ') ;
	np.savetxt('meanVelocity.txt', velocityMean, fmt='%1.5f',delimiter='    ') ;
	np.savetxt('standartDeviationVelocity.txt', velocityStandarDeviation, fmt='%1.5f',delimiter='    ') ;
	np.savetxt('velocityForY.txt', velocityForY, fmt='%1.5f',delimiter='    ') ;

			# Plot Data
				# Ploting
	plt.figure(20+ variableFile.n_files)

	fig,ax1 = plt.subplots()	
	ax1.errorbar(distance[:,1]*1e-6, abs(velocityMean)*5.86e-6/10*distance[:,2], yerr=2*5.86e-6/10*distance[:,2],
				lineStyle ='none', marker='o', color='blue', label ='mean velocity(m.s$^{-1}$) of the RBCs') ;
	plt.xlabel('x position (m)', size=20)
	plt.ylabel('mean velocity (m.s$^{-1}$) of the RBCs ', size=20)
	plt.ticklabel_format(style='sci', axis=('y'), scilimits=(0,0))
	plt.tick_params(axis='y', color='blue', labelsize=15)
		

	ax2 = ax1.twinx()
	ax2.errorbar(distance[:,1]*1e-6, velocityStandarDeviation*5.86e-6/10*distance[:,2],  yerr=1.4*5.86e-6/10*distance[:,2],
				lineStyle='none', marker='^', color='red', label = 'standard deviation (m.s$^{-1}$) of the RBCs' )
	plt.ylabel('standard deviation (m.s$^{-1}$) of the RBCs', size=20)	
	plt.ticklabel_format(style='sci', axis=('y'), scilimits=(0,0))
	plt.tick_params(axis='y', color='red', labelsize=15)
	ax2.spines['right'].set_color('red')
	ax2.spines['left'].set_color('blue')

		
	for i in range(line) :
		plt.figure(30+variableFile.n_files)
		plt.plot(distance[:,1]*1e-6, velocityForY[:,i], 'o', color=variableFile.color_file_video[i])



	return()
