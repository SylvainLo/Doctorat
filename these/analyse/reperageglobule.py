import numpy as np
import cv2

import backgroungstudy



#############################
######### distance  #########
#############################
def distance(x, y, wallTop, wallBot):
	allDisTop = np.sqrt( (wallTop[:, 0]-x)**2 + (wallTop[:, 1]-y)**2 )
	allDisBot = np.sqrt( (wallBot[:, 0]-x)**2 + (wallBot[:, 1]-y)**2 )

	dTop = allDisTop[np.argmin(allDisTop)]
	dBot = allDisBot[np.argmin(allDisBot)]

	return dTop, dBot




#######################################
######### Particle detection  #########
#######################################
def particledetection(nFilm, nameVideoData, subAnswer) :
	### Opening of the avi
	movie = cv2.VideoCapture(nameVideoData[nFilm]) ; print('	', nameVideoData[nFilm])


	### Constants of the video
	movieWidth = int(movie.get(3))
	movieHeight = int(movie.get(4))

	dataDetection = np.zeros((50000,11), dtype=np.float)


	### Detection edges
		# Background detection
	(background) = backgroungstudy.backgroundcreation(movie, movieHeight, movieWidth)
	print('		background ok for video ', nameVideoData[nFilm])

		# Detection edges
	(edgeTop, edgeBot) = backgroungstudy.detectionedges(background, movieWidth, movieHeight)
	print('		edges ok for video ', nameVideoData[nFilm])


	### Kernel and class background
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	backgroundMask = cv2.createBackgroundSubtractorMOG2(varThreshold=10, detectShadows=False)
	backMask = np.zeros((movieHeight, movieWidth), dtype=np.int)


	### Lecture Image and Particle Detecting
		# setting movie
	movie.set(1,1) ; i = int(1)  # place the lecture of the film at second picture (1=option de set, 2=second image)
	nParticle = 0
		
		# Opening of the pictures
	while movie.isOpened():
		ret, frame = movie.read()
		if ret is False:  # if no more frame then stop
			break
		i += 1
			
		# Subtract background
			# Subtract background wit MOG2
		picWithoutBack = backgroundMask.apply(frame)
		backMask[:, :] = 0

			# Subtract background and keep a gray picture
		picSubBack = np.array(cv2.subtract(background,frame[:,:,0]), dtype = np.uint8)
		backMask[picWithoutBack>100] = 1
		picSubMOG2Gray = np.array(picSubBack*backMask, dtype=np.uint8)

		# Verification de condition
		if np.max(picSubMOG2Gray)>15 and i>20 :

			# Modification of the pictures
			picSubMOG2Gray[picSubMOG2Gray<5] = 0
			picEquaHist = cv2.equalizeHist(picSubMOG2Gray)
			picBlur = cv2.medianBlur(picEquaHist, 3)
			ret, picThresh = cv2.threshold(picBlur, 10, 255, cv2.THRESH_BINARY)
			picOpen = cv2.morphologyEx(picThresh, cv2.MORPH_CLOSE, kernel)

			# Particle Detection
				# Contour
			#im2, contours, hierarchy = cv2.findContours(picOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			contours, hierarchy = cv2.findContours(picOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


			if not contours :
				pass
					
			else :
				for con in contours :
					area = cv2.contourArea(con)
					x,y,w,h = cv2.boundingRect(con)
					pointTop = tuple( con[con[:, :, 1].argmin()][0] )
					pointBot = tuple( con[con[:, :, 1].argmax()][0] )
				
					# Limitation des contours
					if x>25 and x+w<movieWidth-25 and 500<area<3000 :
						ell = cv2.fitEllipse(con)
						xG = ell[0][0] ; yG = ell[0][1]
						L = ell[1][0]/2 ; l = ell[1][1]/2
						angle = 90 - ell[2]

					# measure distance Wall/Rbc
						# Distance from the top and bottom wall 
						dG = distance(xG, yG, edgeTop, edgeBot)  
						dTop = distance(pointTop[0], pointTop[1], edgeTop, edgeBot)
						dBot = distance(pointBot[0], pointBot[1], edgeTop, edgeBot)
						
						# Selection of the distance of interest
						dataDetection[nParticle, :] = np.array( [nParticle, i, xG, yG, dG[0], dG[1], dTop[0], dBot[1], L, l, angle] )
						nParticle += 1

					# Movie Observation
					if subAnswer == '1' and i < 1000 :
						cv2.imshow('Initial frame', frame) ; cv2.waitKey(1)
						cv2.imshow('Equalized histogram', picEquaHist) ; cv2.waitKey(1)
						cv2.imshow('Blur', picBlur) ; cv2.waitKey(1)
						cv2.imshow('Threshold', picThresh) ; cv2.waitKey(1)
						cv2.imshow('Opening', picOpen) ; cv2.waitKey(1)
						cv2.drawContours(frame, con, -1, 0, thickness=1) ; cv2.imshow('Contour', frame) ; 	cv2.waitKey(1)

					

	### Save files
		# Reduction files
	dataDetection = np.compress(dataDetection[:,1] != 0, dataDetection, axis=0)

		# if the small distance is bot then necessity to change column
	if np.mean(dataDetection[:,4]) > np.mean(dataDetection[:,5]) :
		partVec = np.array([dataDetection[:,5],dataDetection[:,7]])
		dataDetection[:,5] = dataDetection[:,4]
		dataDetection[:,7] = dataDetection[:,6]
		dataDetection[:,4] = partVec[0,:]
		dataDetection[:,6] = partVec[1,:]

		# Saving data
	distanceFile = 'distanceDataFile_%03d.txt'%(nFilm+1)
	np.savetxt(distanceFile, dataDetection, fmt = '%.0d %.0d %.8e %.8e %.4f %.4f %.4f %.4f %.4f %.4f %.2f' , delimiter='	',
			   header="nParticle, nFrame, pos x , pos y , small d, big d, d Wall//Wall, D Wall//Wall bot, big radius, small radius, angle")
	print('		file of RBC position saved for video ', nameVideoData[nFilm])


	return













## draw point inside contour
#conReshape = np.reshape(con[:][:][:], (-1,2))
#blackFrame = np.zeros((movieHeight,movieWidth), dtype=np.uint8) ; 
#blackFrame = cv2.drawContours(blackFrame, con[:][:][:], -1, 255, cv2.FILLED) ;

## measure center of mass
#pInCont = np.argwhere(blackFrame==255) ;
#m00 = np.sum(np.abs(frame[pInCont[:,0],pInCont[:,1],0]-255)) ; 
#m10 = np.sum(np.abs(frame[pInCont[:,0],pInCont[:,1],0]-255) * pInCont[:,1]+0.5) ;  
#m01 = np.sum(np.abs(frame[pInCont[:,0],pInCont[:,1],0]-255) * pInCont[:,0]+0.5) ;
#X = m10/m00 ; Y = m01/m00 ;

#M = cv2.moments(con) ; cX = (M["m10"] / M["m00"]) ; cY = (M["m01"] / M["m00"]) ;
