import os ;
import fonctionbase ;

import numpy as np ;

import math ;

import matplotlib.pyplot as plt ;



#####################################################
######### Mesure of the lenght of the pulse #########
#####################################################

	######### Form pulse #########

def lenghtpulseforme(nameFile, numeroFile, variableFile) :
	# Loading data
	localData = np.loadtxt(nameFile[numeroFile]) ;

	# Reduction
	interspace = localData[1:,1]-localData[:-1,1] ;
	(vectorTest,) = np.where(interspace > variableFile.interspace_pulse) ;
	(lenght, ) = np.shape(vectorTest) ;
	print(nameFile[numeroFile], ',', vectorTest ) ;

	limMin = np.where(vectorTest < variableFile.max_choix) ;
	if np.size(limMin)==0 :
		pass
	elif np.size(limMin)==1 :
		localData = np.delete(localData, np.arange(0,vectorTest[limMin]+1), axis=0) ;
	else :
		maxLimMin = np.max(limMin)
		localData = np.delete(localData, np.arange(0,vectorTest[maxLimMin]+1), axis=0) ; 
	
	interspace = localData[1:,1]-localData[:-1,1] ;
	(vectorTest,) = np.where(interspace > variableFile.interspace_pulse) ; 
	(lenght, ) = np.shape(vectorTest) ;

	(lineDataNbrRbc,column) = np.shape(localData) ;
	retard = np.zeros(lineDataNbrRbc) ;

	while lenght > variableFile.distance[numeroFile, 2]-1 :  
		massVectorTest = np.zeros(lenght) ;
		
		for i in range(lenght) :
			positionImage = localData[vectorTest[i]+1,1]
			massVectorTest[i] = np.sum((localData[:,1]>positionImage) & (localData[:,1]<positionImage+10)) ;

		minToSupress = np.argmin(massVectorTest) ;
		vectorTest = np.delete(vectorTest, minToSupress) ;
		(lenght, ) = np.shape(vectorTest) ;
	
	print(nameFile[numeroFile], ',', vectorTest ) ;

	for i in range( lenght+1 ) :
		if i == 0 :
			retard[0:vectorTest[i]] = localData[0:vectorTest[i],1] - np.min(localData[0:vectorTest[i],1]) ;
		elif i > 0 and i < lenght  :
			retard[vectorTest[i-1]+1:vectorTest[i]] = localData[vectorTest[i-1]+1:vectorTest[i],1] - np.min(localData[vectorTest[i-1]+1:vectorTest[i],1]) ;
		else :
			retard[vectorTest[i-1]+1:] = localData[vectorTest[i-1]+1:,1] - np.min(localData[vectorTest[i-1]+1:,1]) ;
	

	# Histogram
	(histTau, binTau) = np.histogram(retard, bins=variableFile.histogramm_bin, range=(0,variableFile.histogramm_range)) ;

		# Saving
	np.savetxt('hisTauLocal_%03d.txt'%(numeroFile+1), histTau, fmt='%1.2f',delimiter='    ')
			
	# Cumsum
	sumTau = np.cumsum(histTau)
	sumTauNormalized = sumTau/np.max(sumTau)

		# Saving
	np.savetxt('sumTauNormalized_%03d.txt'%(numeroFile+1), sumTauNormalized, fmt='%1.2f',delimiter='    ')

	# Lenght Pulse
	lenghtPulse = np.zeros(len(variableFile.limit_pulse))
	for j in range(len(variableFile.limit_pulse)) :
		lenghtPulse[j] = np.where(sumTauNormalized>variableFile.limit_pulse[j])[0][0] ;
			
		# saving
	np.savetxt('localLenghtPulse_%03d.txt'%(numeroFile+1), lenghtPulse, fmt='%1.5f',delimiter='    ')

	return()






	######### Function plotformepulse #########

def plotformepulse(subDirectoryData, variableFile) :
	### Plot Local variable ###
		# Plot data histo
			# Searching data histo #
	(nameDataToPlot, sizeDataToPlot) = fonctionbase.filestoanalyse(subDirectoryData, '*hisTauLocal*') ;

			# Prepare plot
	nbrRow = math.ceil(sizeDataToPlot/3) ;
	plt.figure(50+variableFile.n_files) ;
		
			# Plot histogram
	for i in range(sizeDataToPlot) :
		histTau = np.loadtxt(nameDataToPlot[i]) ;

		plt.subplot(nbrRow, 3, i+1)
		plt.plot(np.linspace(0,variableFile.histogramm_range,variableFile.histogramm_bin),histTau)


		# Plot data cumsum
			# Searching data Cumsum #
	(nameDataToPlot, sizeDataToPlot) = fonctionbase.filestoanalyse(subDirectoryData, '*sumTauNormalized*') ;

			# Prepare plot
	plt.figure(60+variableFile.n_files) ;
		
				# Plot curve sum
	for i in range(sizeDataToPlot) :
		sumTauNormalized = np.loadtxt(nameDataToPlot[i]) ;

		plt.plot(np.linspace(0,variableFile.histogramm_range,variableFile.histogramm_bin),sumTauNormalized, color=variableFile.color_file_video[i])


	### Plot Global Data ###
		# Searching local Data
	(nameDataLocal, sizeDataLocal) = fonctionbase.filestoanalyse(subDirectoryData, '*localLenghtPulse*') ;
	lenghtGlobalData = np.array([])

		# Creating global Data
	for i in range(sizeDataLocal) :
		localDataLoaded = np.loadtxt(nameDataLocal[i]) ; 
		lenghtGlobalData = np.append(lenghtGlobalData, localDataLoaded)

	lenghtGlobalData = lenghtGlobalData.reshape(sizeDataLocal,-1)

		# Saving data

	np.savetxt('globalLenghtData.txt', lenghtGlobalData, fmt='%1.3f',delimiter='    ')

		# Plot Data
			# Ploting
		#plt.figure(70+variableFile.n_files)
		#for i in range(cte) :		
		#	plt.plot(distance[:,1], lenghtGlobalData[:,i], 'o', color=colorFileVideo[i]) ;

	return()
