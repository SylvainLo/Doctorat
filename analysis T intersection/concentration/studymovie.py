import numpy as np
import cv2
import math

import transformation_picture





###############################################
######### Selection part of interest  #########
###############################################
def crop(img) :
	"""
	:param img: image in gray level
 	manual crop of the picture
	"""

	### function for trackbar
	def nothing(x) :
		pass

	### variable and windows
		# variable
	l, c = np.shape(img)
	imgCopy = np.copy(img)

		# Create windows
	cv2.namedWindow('crop')
	cv2.createTrackbar( 'y top', 'crop', 0, l, nothing)
	cv2.createTrackbar( 'y bot', 'crop', l, l, nothing )
	cv2.createTrackbar( 'x left', 'crop', 0, c, nothing )
	cv2.createTrackbar( 'x right', 'crop', c, c, nothing )
	cv2.createTrackbar( '0 : OFF \n1 : ON', 'crop', 0, 1, nothing )


		# creation windows
	while 1:
		cv2.imshow('crop', imgCopy) ; cv2.waitKey(1)

		y1 = cv2.getTrackbarPos( 'y top', 'crop' )
		y2 = cv2.getTrackbarPos( 'y bot', 'crop' )
		x1 = cv2.getTrackbarPos( 'x left', 'crop' )
		x2 = cv2.getTrackbarPos( 'x right', 'crop' )
		g = cv2.getTrackbarPos( '0 : OFF \n1 : ON', 'crop' )

		imgCopy = np.copy( img )
		cv2.rectangle( imgCopy, (x1, y1), (x2, y2), 255, 1 )

		if g == 1 :
			lim = [y1, y2, x1, x2]
			break

	cv2.destroyAllWindows( )

	return lim





######################################
######### Velocity analysis  #########
######################################
def velocityanalysis(img1, img2, lim) :
	# Match template
	box, corResult = transformation_picture.correlationimage( img1[lim[0]:lim[1], :],
																  img2[lim[0]:lim[1], :] )
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc( corResult )

	# Velocity
	hBox, wBox = abs( box[3]-box[1] ), abs( box[2]-box[0] )
	max_loc = (max_loc[0] + wBox/2, max_loc[1] + hBox/2)
	v = math.sqrt( (max_loc[0] - (box[0]+box[2])/2 )**2+(max_loc[1]-(box[1]+box[3]) / 2)**2 )

	# Save velocity
	print( '			v = ', v )

	return v




###############################################
######### Selection part of interest  #########
###############################################
def selectionline(img) :
	"""
	manuel selection of profile intensity line
	img in Gray (one channel for color)
	"""

	### function for trackbar
	def nothing(x) :
		pass

	### variable and windows
		# variable
	l, c = np.shape(img)
	imgCopy = np.copy(img)

		# Create windows
	cv2.namedWindow('line', cv2.WINDOW_AUTOSIZE)
	cv2.resizeWindow( 'line', int(c/2), int(l/2) )
	cv2.createTrackbar('y', 'line', 0, c, nothing)

		# creation windows
	while 1:
		cv2.imshow('line', imgCopy) ; k = cv2.waitKey(1) & 0xFF

		y = cv2.getTrackbarPos('y', 'line')

		imgCopy = np.copy( img )
		cv2.line( imgCopy, (y,0), (y,l), 0, 1 )

		if k == 13 : 																					  # 13 for enter
			yLine = y
			break

	cv2.destroyAllWindows( )

	return yLine