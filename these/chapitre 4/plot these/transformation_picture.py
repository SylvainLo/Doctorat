import cv2
import numpy as np

import matplotlib.pyplot as plt
import scipy.optimize as sci
import scipy.interpolate as sci2

import time
import math



########################################
######### Background creation  #########
########################################
def backgroundcreation(movie):
	"""
	maximal intensity of all the frame
	:param movie: a cv2 object
	"""

	### Constant
	movie.set(cv2.CAP_PROP_POS_FRAMES, 0)           													  # base index 0
	w = int( movie.get(cv2.CAP_PROP_FRAME_WIDTH) )
	h = int( movie.get(cv2.CAP_PROP_FRAME_HEIGHT) )

	backMaxInt = np.zeros((h, w), dtype=np.uint8)
	sumMov = np.zeros((h, w), dtype=np.int32)
	i = int(0)


	### Read Movie
	while movie.isOpened():

		# Opening picture
		ret, frameForBackground = movie.read()
		if ret is False :  # if no more frame then stop
			break
		frameForBackground = frameForBackground[:, :, 0]

		# creation mask to take the maximal intensity
		mask = backMaxInt < frameForBackground
		backMaxInt[mask] = frameForBackground[mask]

		# sum picture
		i += 1
		sumMov = np.add(sumMov, frameForBackground)


	### Mean Intensity movie
	backMean = np.array(sumMov/i, np.uint8)
	cv2.imshow( 'mean intensity', backMean ) ; cv2.waitKey( 10 )
	cv2.imshow( 'max intensity', backMaxInt ) ; cv2.waitKey( 10 )

	### Threshold
	intMean, intStd = threshbackground(backMean)
	lim = math.ceil(intMean)

	mask = backMaxInt>lim
	background = np.copy(backMaxInt)
	background[mask] = lim
	print('value threshold = ', lim)

	cv2.imshow( 'background', background ) ; cv2.waitKey( 10 )

	return background, backMean, backMaxInt



drag = None
sol = (0,0,0,0)
backstage = np.array([], dtype=np.uint8)
kk = 0

def onmouse1( event, x, y, flags, param ) :
	global drag, sol, k, backstage

	if event == cv2.EVENT_LBUTTONDOWN :
		drag = x, y
		sol = 0, 0, 0, 0

	elif event == cv2.EVENT_LBUTTONUP :
		if sol[2] > sol[0] and sol[3] > sol[1] :
			backstage = param[sol[1]:sol[3], sol[0]:sol[2]]

			cv2.imshow( "result", backstage)
			k = 1

		else :
			drag = None

	elif drag :
		if flags & cv2.EVENT_FLAG_LBUTTON :
			minpos = min( drag[0], x ), min( drag[1], y )
			maxpos = max( drag[0], x ), max( drag[1], y )
			sol = minpos[0], minpos[1], maxpos[0], maxpos[1]
			img = cv2.cvtColor( param, cv2.COLOR_GRAY2BGR )
			cv2.rectangle( img, (sol[0], sol[1]), (sol[2], sol[3]), (0, 255, 255), 1 )
			cv2.imshow( "threshold background", img )

		else :
			drag = None


def threshbackground(img) :
	cv2.namedWindow("threshold background", 1)
	cv2.setMouseCallback("threshold background", onmouse1, param=img )

	while 1 :
		cv2.imshow("threshold background", img)
		k = cv2.waitKey(1)
		if k == 27 or 0xFF == ord("q") or k == 1 :
			break

	cv2.destroyAllWindows( )

	valMean = np.mean(backstage)
	valStd = np.std(backstage)
	print(valMean, valStd)

	return valMean, valStd




#############################################
######### Detection edges et masks  #########
#############################################
def detectionedges( img ) :
	### value picture
		# size picture
	l, c = np.shape(img)

		# kernel for erosion
	kernelEdges = np.ones( (1, 10), np.uint8 )


	### Bluring
	imgCopy = np.copy( img )
	imgMedFilt = cv2.medianBlur( imgCopy, 3 )


	### Gradient image
	#grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
	grad = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
	gradAbsFloat = np.absolute(grad)
	gradAbs = np.uint8(gradAbsFloat)
	cv2.imshow( 'gradient', gradAbs ) ; cv2.waitKey( 100 )

	### Threshold
		# Create windows
	cv2.namedWindow( 'threshold', cv2.WINDOW_AUTOSIZE )
	cv2.createTrackbar( 'Threshold', 'threshold', 0, 255, nothing )
	cv2.createTrackbar( '0 : OFF \n1 : ON', 'threshold', 0, 1, nothing )

		# Create edges vector
	edges = np.zeros( (l, c), dtype=np.uint8 )

		# Threshold by user
	while 1 :
		cv2.imshow( 'threshold', edges ) ; cv2.waitKey( 10 )

		k = cv2.getTrackbarPos( '0 : OFF \n1 : ON', 'threshold' )
		if k == 1 :
			break

		treshValue = cv2.getTrackbarPos( 'Threshold', 'threshold' )
		pet, edges = cv2.threshold( gradAbs, treshValue, 255, cv2.THRESH_BINARY)

		### Erosion
		edgesEroded = cv2.erode( edges, kernelEdges, iterations=1 )
		cv2.imshow( 'edgesEroded', edgesEroded ) ; cv2.waitKey( 100 )

		### dataEdges
		dataEdges = np.argwhere( edgesEroded > 0.5 )

		if np.size(dataEdges) > 10 :
		### separation of the 2 edges
			imgCopy2 = imgMedFilt.copy()
			meanDataEdges = np.mean( dataEdges[:, 0] )
			x = np.linspace( 0, c-1, c, dtype=np.int16 )

			# Superior edges
			datEdgesPlus = np.compress( dataEdges[:, 0] < meanDataEdges, dataEdges, axis=0 )
			if np.size(datEdgesPlus) :
				popt, pcov = sci.curve_fit( fct, datEdgesPlus[:, 1], datEdgesPlus[:, 0] )
				datEdPlus = np.array( fct( x, popt[0], popt[1] ), dtype=np.int16 )

			# Inferior edges
			datEdgesMinus = np.compress( dataEdges[:, 0] > meanDataEdges, dataEdges, axis=0 )
			if np.size(datEdgesMinus) :
				popt, pcov = sci.curve_fit( fct, datEdgesMinus[:, 1], datEdgesMinus[:, 0] )
				datEdMinus = np.array( fct( x, popt[0], popt[1] ), dtype=np.int16 )

			imgCopy2[datEdPlus, x] =0 ; imgCopy2[datEdMinus, x] = 0
			cv2.imshow( 'wall', imgCopy2) ; cv2.waitKey( 100 )


	# Close windows
	cv2.destroyAllWindows( )

	return datEdPlus, datEdMinus


def nothing(x) :
	pass


def fct(x, a, b) :
	return a*x + b







###############################################
######### Function correlation image  #########
###############################################
drag_start = None
sel = (0,0,0,0)
resultN = np.array([], dtype=np.uint8)
k = 0


def onmouse( event, x, y, flags, param ) :
	global drag_start, sel, resultN, k

	if event == cv2.EVENT_LBUTTONDOWN :
		drag_start = x, y
		sel = 0, 0, 0, 0

	elif event == cv2.EVENT_LBUTTONUP :
		if sel[2] > sel[0] and sel[3] > sel[1] :
			patch = param[0][sel[1] :sel[3], sel[0] :sel[2]]

			result = cv2.matchTemplate( param[1], patch, cv2.TM_CCOEFF_NORMED )
			result = np.abs(result)**3

			val, result = cv2.threshold( result, 0.01, 0, cv2.THRESH_TOZERO )
			resultN = cv2.normalize( result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U )
			cv2.imshow( "result", resultN )
			k = 1

		else :
			drag_start = None

	elif drag_start :
		if flags & cv2.EVENT_FLAG_LBUTTON :
			minpos = min( drag_start[0], x ), min( drag_start[1], y )
			maxpos = max( drag_start[0], x ), max( drag_start[1], y )
			sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
			img = cv2.cvtColor( param[0], cv2.COLOR_GRAY2BGR )
			cv2.rectangle( img, (sel[0], sel[1]), (sel[2], sel[3]), (0, 255, 255), 1 )
			cv2.imshow( "correlation", img )

		else :
			drag_start = None



def correlationimage(img1, img2) :
	cv2.namedWindow("correlation", 1)
	cv2.setMouseCallback("correlation", onmouse, param=[img1, img2] )

	while 1 :
		cv2.imshow("correlation", img1)
		k = cv2.waitKey(1)
		if k == 27 or 0xFF == ord("q") or k == 1 :
			break

	cv2.destroyAllWindows( )

	return sel, resultN





########################################
######### Background creation  #########
########################################
def lineintensity(dat, back, posY, edT, edB) :
	### pre variable
		# value proportionnal to concentration
	divide = np.divide(dat, back, dtype=np.float64)
	divide[divide<1] = 1
	hemato = np.log( divide )

		# width between edges along x
	diff = abs(np.subtract(edT, edB))
	yInt = np.linspace(0,1,201)


	### value intensity for each line
	hem = np.zeros(201)

		# interpolation and mean over 5 column
	for p in range(-20,21,1) :
		y = posY + p
		val = hemato[edT[y]:edB[y], y]
		xVal = np.linspace(0, 1, diff[y])

		# interpolation
		f = sci2.interp1d(xVal, val, kind='linear', copy=True)
		hemLine = f(yInt)

		# add hematocrit of each line
		hem += hemLine

		# mean hematocrit over line
	hem = hem/41


	return hem





########################################
######### Background creation  #########
########################################
def extractbackground(movie):
	"""
	maximal intensity of all the frame
	:param movie: a cv2 object
	"""

	### Constant
	movie.set(cv2.CAP_PROP_POS_FRAMES, 0)           													  # base index 0
	w = int( movie.get(cv2.CAP_PROP_FRAME_WIDTH) )
	h = int( movie.get(cv2.CAP_PROP_FRAME_HEIGHT) )

	back = np.zeros((h, w), dtype=np.float64)
	maskDiv = np.ones((h, w), dtype=np.float64)
	fgmask2 = np.ones((h, w), dtype=np.float64)

	fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=10, detectShadows=False)

	### Read Movie
	while movie.isOpened():

		# Opening picture
		ret, frame = movie.read()
		if ret is False :  # if no more frame then stop
			break

		frame = frame[:, :, 0]

		# creation mask to take the maximal intensity
		fgmask = fgbg.apply(frame)
		fgmask = fgmask/255
		fgmask2[fgmask==1] = 0 ; fgmask2[fgmask==0] = 1

		# creation mask to take the minimal intensity
		back = back + frame*fgmask2
		maskDiv= maskDiv + fgmask2


	### Mean Intensity movie
	print('back', back)
	print('maskDiv', maskDiv)

	backdoor = back/maskDiv
	backdoor2 = np.array(backdoor, dtype=np.uint8)

	plt.figure( 'test' )
	plt.imshow( backdoor )
	plt.figure( 'test2' )
	plt.imshow( backdoor2 )
	plt.show()

	return backdoor