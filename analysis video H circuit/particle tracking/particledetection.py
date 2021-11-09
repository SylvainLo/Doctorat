import cv2 ;

import numpy as np ;
import pandas ;

#import trackpy ;



#######################################
######### Particle detection  #########
#######################################

	######### Function Background Image #########

def backgroundimage(movie, h, movieHeight, movieWidth, limitIntensity) :
	# Ouverture first picture
	movie.set(1,1)                                                 # place the lecture of the film at first picture (1=option de set, 0=first image)
	ret, frame = movie.read() ; frameBackground = np.copy(frame[:,:,0])

	# Edges
		# Canny operator
	frameBackground[frameBackground>limitIntensity] = 255
	#cv2.imshow('frameBackground', frameBackground) ; cv2.waitKey(10000) ;
	edges = cv2.Canny(frameBackground, 30, 150, apertureSize = 3)
	#cv2.imshow('edges', edges) ; cv2.waitKey(10000) ;

		# Growth of edges to reduce the effect of noises
	newEdges = np.zeros((movieHeight,movieWidth),dtype=bool) ;
	for u in range(14,movieWidth-15) :
		for v in range(9,movieHeight-10) :
			if edges[v,u] == 255 :
				newEdges[v-10:v+10,u-15:u+15] = 1
	#plt.imshow(newEdges) ; plt.show()

		# Search gravity point of the edges
	yline = np.arange(movieHeight) + 1 ; ysum = 0 ; sumNewEdges = np.sum(newEdges, dtype=int) ;
	for s in range(movieWidth) :
		ysum = ysum + np.sum( yline*newEdges[:,s] )
	gravityEdges = int(ysum/(sumNewEdges)) ; print('	gravityEdges = ', gravityEdges)

	# Final transformation (inversion intensity)
	backgroundReduced = frame[gravityEdges-h:gravityEdges+h,:,0]
	print('		background have been calculated')

	return (backgroundReduced, gravityEdges)





	######### Function Particle detection simple (only the number of Rbc) or complex (the position, mass...)) #########

def particledetection(nFilm, nameVideoData, variableFile) :			
	# Opening of the avi
	movie = cv2.VideoCapture(nameVideoData[nFilm]) ; print('	', nameVideoData[nFilm]) ;

	# Constantes of the video
		# Constante movies
	movieWidth = int(movie.get(3)) ; movieHeight = int(movie.get(4)) ;
	movieFps = movie.get(5) ; movieNbrFrames = int(movie.get(7)) ; movieFormat = movie.get(9) ;
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

		# Background
	(backgroundImage, gravityEdges) = backgroundimage(movie, variableFile.reduction_picture, movieHeight, movieWidth, variableFile.limit_Intensity) ;
	(heightShape,widthShape) = np.shape(backgroundImage) ; 
	backgroundImage = np.array(backgroundImage, dtype = np.int16) ;

		# Saving general parameter of the video
	savePropertyMovie = np.array([movieNbrFrames, movieFps, movieWidth, movieHeight, 
									   variableFile.reduction_picture, gravityEdges]) ;
	namePropertyMovie = 'property-movie_%03d.txt'%(nFilm+1) ;
	np.savetxt(namePropertyMovie, savePropertyMovie[None, :], delimiter='    ') ;

	# Lecture Image and Particle Detecting
	movie.set(1,2) ; i = 1 ;                  # place the lecture of the film at second picture (1=option de set, 2=second image)
				
		# Constantes particle detecting
	columnDetection = ['x', 'y', 'mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'nParticle', 'frame']; 
	dataFrameDetection = pandas.DataFrame(columns=columnDetection) ;

		# Opening of the pictures
	movie.set(1,2) ; i = 1 ;                  # place the lecture of the film at second picture (1=option de set, 2=second image)
	while (movie.isOpened()):
		ret, frame = movie.read()
		i += 1 ; #print('i = ', i)

		if ret == False:  # if no more frame then stop 
			break

			# Reduction of the pictures
		frameRediuced = frame[gravityEdges-variableFile.reduction_picture:gravityEdges+variableFile.reduction_picture, :, 0] ;	# change size image
		frameRediuced = np.array(frameRediuced, dtype=np.int16) ;

			# Substract of the background
		frameWithoutBackground = np.abs(np.subtract(frameRediuced, backgroundImage)) ; 
		frameWithoutBackground = np.array(frameWithoutBackground, dtype=np.uint8) ;				# substract background	

			# search for the maximal intensity
		maxIntensity = frameWithoutBackground.max() ;

			# Particle Detecting
		if maxIntensity > 30 :
				# Threshold and erosion
			frameWithoutBackground[frameWithoutBackground < 20] = 0 ;
			frameEroded = cv2.erode(frameWithoutBackground, kernel) ;
			#cv2.imshow('frameEroded', frameEroded) ; cv2.waitKey(100) ;

				# Detector of the particles
			f = trackpy.locate(frameEroded, 9, minmass=200, invert=False, percentile = 20) ;
			f['frame'] = i ;
			#plt.figure() ;
			#trackpy.annotate(f, frameEroded) ;

				# Adding to the data
			newDataFrame = [dataFrameDetection, f] ; dataFrameDetection = pandas.concat(newDataFrame) ;

	# Saving the data
		# Adding an index to pandas
	(rowDetection, column) = np.shape(dataFrameDetection) ;
	indexDetection = np.linspace(0, rowDetection-1, rowDetection, dtype=int) ; dataFrameDetection = dataFrameDetection.set_index(indexDetection) ;
	dataFrameDetection['nParticle'] = dataFrameDetection.index ;
				
		# Saving the Data
	dataFrameDetectionFile = 'dataFrameDetectionVelocityFile_%03d.csv'%(nFilm+1) ;
	dataFrameDetection.to_csv(dataFrameDetectionFile)
	print('files have been saved')

	return()
