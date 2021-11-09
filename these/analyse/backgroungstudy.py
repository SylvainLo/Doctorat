import numpy as np ;				#print('numpy imported')
import scipy.optimize as sci ;

import cv2 ;						#print('cv2 imported')






########################################
######### Background creation  #########
########################################
def backgroundcreation(movie, movieHeight, movieWidth) :
	### Constante
	movie.set(1,2) ; i = 0 ; 
	backgroundSum = np.zeros((movieHeight, movieWidth), dtype=np.int32)

	### Read Movie
	while (movie.isOpened()) :

		# Opening picture
		ret, frameForBackground = movie.read()
		if ret == False or i>1000 :  # if no more frame then stop 
			break
		frameForBackground = frameForBackground[:,:,0] ;

		# Sum of the picture
		i += 1 ; #print(i)
		backgroundSum = np.add(backgroundSum, frameForBackground) ;

	### Background
	background = np.array(backgroundSum/i, np.uint8)
	cv2.imshow('background', background) ; cv2.waitKey(1000) ;

	return(background)




##########################################
######### Function for Trackbar  #########
##########################################
def nothing(x) :
    pass




########################################
######### Fit affine function  #########
########################################

def fct(x, a, b) :
	return a*x + b





#############################################
######### Detection edges et masks  #########
#############################################
def detectionedges(img, movieWidth, movieHeight) :
	### Bluring
	backgroundMedianFiltering = cv2.medianBlur(img, 3)

	### Threshold
		# Create windows
	cv2.namedWindow('threshold')
	cv2.createTrackbar('Threshold','threshold',0,255,nothing)
	cv2.createTrackbar('0 : OFF \n1 : ON', 'threshold',0,1,nothing)

		# Create edges vector
	edges = np.zeros((movieHeight,movieWidth), dtype=np.uint8) ;

		# Threshold by user
	while(1) :
		cv2.imshow('threshold', edges) ; cv2.waitKey(10) ;
		k = cv2.getTrackbarPos('0 : OFF \n1 : ON', 'threshold')
		if k == 1 :
			break
		tresh = cv2.getTrackbarPos('Threshold', 'threshold')
		pet, edges = cv2.threshold(backgroundMedianFiltering, tresh, 255, cv2.THRESH_BINARY_INV)

		# Close windows
	cv2.destroyAllWindows()

	### Erosion
	kernelEdges = np.ones((2,10),np.uint8)
	edgesEroded = cv2.erode(edges,kernelEdges,iterations = 1)
	cv2.imshow('edgesEroded', edgesEroded) ; cv2.waitKey(100) ;

	### dataEdges
	dataEdges = np.argwhere(edgesEroded == np.max(edgesEroded)) ;

	### separation of the 2 edges
	meanDataEges = np.mean(dataEdges[:,0]) ; 
	x = np.linspace(0,movieWidth+1,movieWidth*5) ;
				
		# Superior edges
	datEdgesPlus = np.compress(dataEdges[:,0]<meanDataEges, dataEdges, axis=0) ;
	popt, pcov =  sci.curve_fit(fct, datEdgesPlus[:,1], datEdgesPlus[:,0]) ;
	datEdPlus = np.empty((np.size(x),2)) ; datEdPlus[:,0] = x ; datEdPlus[:,1] = fct(x, popt[0], popt[1]) ;
	
		# Inferior edges
	datEdgesMinus = np.compress(dataEdges[:,0]>meanDataEges, dataEdges, axis=0) ; 
	popt, pcov =  sci.curve_fit(fct, datEdgesMinus[:,1], datEdgesMinus[:,0]) ;
	datEdMinus = np.empty((np.size(x),2)) ; datEdMinus[:,0] = x ; datEdMinus[:,1] = fct(x, popt[0], popt[1]) ;
	

	return(datEdPlus, datEdMinus)

