import numpy as np
import cv2
import math

import transformation_picture





###############################################
######### Selection part of interest  #########
###############################################
def selectionpart(img, nbr) :
	"""
	manuel detection of the channel
	img in Gray (one channel for color)
	nbr numero of the video
	"""

	### function for trackbar
	def nothing(x) :
		pass

	### variable and windows
		# variable
	l, c = np.shape(img)
	imgCopy = np.copy(img)
	lim = np.array([[]], dtype=np.int) ; i = int(1)

		# Create windows
	cv2.namedWindow('crop', cv2.WINDOW_NORMAL)
	cv2.createTrackbar('x left', 'crop', 0, c, nothing)
	cv2.createTrackbar('x right', 'crop', 0, c, nothing )

		# creation windows
	while 1:
		cv2.imshow('crop', imgCopy)
		k = cv2.waitKey(1) & 0xFF

		x1 = cv2.getTrackbarPos( 'x left', 'crop' )
		x2 = cv2.getTrackbarPos( 'x right', 'crop' )

		imgCopy = np.copy( img )
		cv2.rectangle( imgCopy, (x1, 0), (x2, l), 255, 1 )

		if k == ord('c') :
			if i == 1 :
				lim = np.array([x1,x2], dtype=np.int)
			else :
				lim = np.concatenate((lim,np.array([x1,x2])), axis=0)
			i += 1
			cv2.rectangle(img, (x1,0), (x2,l), 0, 2)

		elif k == 27:
			break

	cv2.destroyAllWindows( )

	### save data
		# save number channel
	nbrChannel = open( 'nbr_channel_analysis_%02d.txt'%nbr, "w" )
	nbrChannel.write( "{}".format( int(i-1) ) )
	nbrChannel.close()

		# save mask
	np.savetxt('mask_channel_%02d.txt'%nbr, lim, fmt='%4d', delimiter='	')

	return lim, i-1








#######################################
######### Analysis intensity  #########
#######################################

def intensityvariation(mov, back) :
	"""
	evolution intensity time
	  -mov, a cv2 object
	  -back, background image in gray (same depth as mov)
	"""

	### Constant
	mov.set(1, 1)
	nbrF = int( mov.get(cv2.CAP_PROP_FRAME_COUNT))

	i = 0
	intensity = np.zeros(nbrF, dtype=np.int32)
	c = np.zeros(nbrF, dtype=np.float)

	backInv = cv2.bitwise_not(back)

	### Read Movie
	while mov.isOpened():

		# Opening picture
		ret, frame = mov.read()
		if ret is False :  # if no more frame then stop
			break
		frame = frame[:, :, 0]

		# Subtract background
			# by dividing (concentration)
		divide = np.divide( cv2.bitwise_not(frame), backInv, dtype=np.float64 )
		divide[divide<1.2] = 1
		hemato = np.log( divide )

			# by subtracting (intensity)
		diff = cv2.subtract(back[:, :], frame)

		# Threshold
		diff[diff<30] = 0

		# Intensity
		intensity[i] = np.sum( diff )
		c[i] = np.mean( hemato )

		# Sum of the picture
		i += 1

	return intensity, c




#################################
######### Unique bolus  #########
#################################
def functionbolus(datI, datC, datBase) :
	"""
	:param datI: total sum intensity
	:param datC: c.log(I/I0)
	:param datBase: value under video
	:return: 2 ndarray reconstruction of a single bolus
	"""

	### variable
	ik = int(0)
	s = np.size(datI)
	datCopy = np.copy(datI)

	period = int(datBase[0,5] * 0.95 * datBase[0,6])
	lim = np.percentile(datI[datI>2000], 10)
	print(lim)

	bolC = np.zeros(period, dtype=np.float)
	bolI = np.zeros(period, dtype=np.float)
	borne = np.array([], dtype=np.int)

	### bolus total
	while np.argmax(datCopy > lim) != 0 :
		a = np.argmax(datCopy > lim)

		if np.all(datCopy[a:a+30]>lim) :
			if a+period < s :
				bolI = bolI + datI[a:a+period]
				bolC = bolC + datC[a:a+period]
				borne = np.append(borne, np.array([a]))
				datCopy[a:a+period] = 0
				ik += 1
			else :
				bolI[0 :s-a-1] = bolI[0 :s-a-1] + datI[a :-1]
				bolC[0 :s-a-1] = bolC[0 :s-a-1]+datC[a :-1]
				borne = np.append( borne, np.array( [a] ) )
				datCopy[a:-1] = 0
				ik += 1

		else :
			datCopy[a] = 0


	bolI = bolI/ik ; bolC = bolC / ik

	return bolI, bolC









######################################
######### Velocity analysis  #########
######################################
def velocityanalysis(mov, c, value, data) :
	# find the two frame of interest
	img1, img2 = transformation_picture.extractpicture(mov, c, data)

	# Match template
	box, corResult = transformation_picture.correlationimage( img1, img2 )
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc( corResult )

	# Velocity
	hBox, wBox = abs( box[3]-box[1] ), abs( box[2]-box[0] )
	max_loc = (max_loc[0] + wBox/2, max_loc[1] + hBox/2)
	v = math.sqrt( (max_loc[0] - (box[0]+box[2])/2 )**2+(max_loc[1]-(box[1]+box[3]) / 2)**2 )

	# Save velocity
	print( '			v = ', v )

	return v

