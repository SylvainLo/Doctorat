import numpy as np ;				#print('numpy imported')
import matplotlib.pyplot as plt ;	#print('matplotlib imported')
import math ;						#print('math imported')
import cv2 ;						#print('cv2 imported')

print('modules have been charged')




#############################################
######### Detection edges et masks  #########
#############################################
def detectionedges(img, movieWidth, movieHeight) :
	# Bluring
	backgroundMedianFiltering = cv2.medianBlur(background, 10)
	cv2.imshow('backgroundMedianFiltering', backgroundMedianFiltering) ; cv2.waitKey(1) ;

	# Canny operator
	edges = cv2.Canny(img, 50, 125, apertureSize = 3) ;
	cv2.imshow('edges', edges) ; cv2.waitKey(1) ;

	# Creating lines
	cteSeparationPicture = math.floor(movieHeight/2) ;
	imgEdgesLine = np.zeros((movieHeight,movieWidth), dtype=np.uint8) ;

	for u in range(2) :	
		dataEdgesPartial = np.argwhere(edges[u*cteSeparationPicture:cteSeparationPicture*(u+1), :] == 255) ;

		[vx,vy,x,y] = cv2.fitLine(dataEdgesPartial, cv2.DIST_L2, 0, 0.01, 0.01) ; print('vx = ', vx, 'vy = ', vy, 'x = ', x, 'y', y) ;
		vx=0 ; vy=1 ;
		lefty = int((-y*vx/vy) + x)+u*cteSeparationPicture ; righty = int(((movieWidth-y)*vx/vy)+x)+u*cteSeparationPicture ;
		
		imgEdgesLine = cv2.line(imgEdgesLine,(0,lefty),(movieWidth,righty), 255, 1) ;
	cv2.imshow('imgEdgesLine', imgEdgesLine) ; cv2.waitKey(1) ;

	# dataEdges
	dataEdges = np.argwhere(imgEdgesLine == 255) ;

	return(edges, dataEdges)
