import cv2
import numpy as np

import scipy.optimize as sci


########################################
######### Background creation  #########
########################################
def backgroundcreation(movie):
	"""
	mean intensity of all the frame

	movie, a cv2 object
	"""

	### Constant
	movie.set(1, 1)
	w = int( movie.get( 3 ) )
	h = int( movie.get( 4 ) )

	i = 0
	backgroundSum = np.zeros((h, w), dtype=np.int32)

	### Read Movie
	while movie.isOpened():

		# Opening picture
		ret, frameForBackground = movie.read()
		if ret is False :  # if no more frame then stop
			break
		frameForBackground = frameForBackground[:, :, 0]

		# Sum of the picture
		i += 1
		backgroundSum = np.add(backgroundSum, frameForBackground)

	### Background
	background = np.array(backgroundSum / i, np.uint8)
	cv2.imshow('background', background) ; cv2.waitKey(10)

	return background







###############################################
######### Fonction correlation image  #########
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

	return (sel, resultN)



def extractpicture(mov, i, data) :
	### value for analysis
		# set movie and take first picture
	mov.set( 1, 1 )
	ret, frame = mov.read( )
	frame = frame[:, :, 0]

		# constant
	period = data[i,6]*data[i,7]

		# Create windows
	cv2.namedWindow( 'correlation for velocity', cv2.WINDOW_AUTOSIZE )
	cv2.createTrackbar( 'frame', 'correlation for velocity', 0, int(period-1), nothing )
	cv2.createTrackbar( '0 : OFF \n1 : ON', 'correlation for velocity', 0, 1, nothing )


	### choose picture to use for correlation
	while 1 :
		cv2.imshow( 'correlation for velocity', frame ) ; cv2.waitKey( 10 )

		k = cv2.getTrackbarPos( '0 : OFF \n1 : ON', 'correlation for velocity' )
		if k == 1 :
			break

		nFr = cv2.getTrackbarPos( 'frame', 'correlation for velocity' )

		mov.set( 1, nFr )
		ret, frame = mov.read( )
		frame = frame[:, :, 0]

		mov.set( 1, nFr+1)
		ret, frame2 = mov.read( )
		frame2 = frame2[:, :, 0]

	# Close windows
	cv2.destroyAllWindows( )

	return frame, frame2




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

			imgCopy2[datEdPlus, x] =255 ; imgCopy2[datEdMinus, x] = 255
			cv2.imshow( 'wall', imgCopy2) ; cv2.waitKey( 100 )


	# Close windows
	cv2.destroyAllWindows( )

	return datEdPlus, datEdMinus


def nothing(x) :
	pass

def fct(x, a, b) :
	return a*x + b
