import numpy as np
import cv2
import math

import transformation_picture

import matplotlib.pyplot as plt




#######################################
######### Analysis intensity  #########
#######################################

def intensityvariation(mov, back, lim) :
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
		frame = frame[lim[0] :lim[1], :, 0]

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
def functionbolus(datI, datC, datBase, ax) :
	"""
	:param datI: total sum intensity
	:param datC: c.log(I/I0)
	:param datBase: value under video
	:param ax: pour le plot
	:return: 2 ndarray reconstruction of a single bolus
	"""

	### variable
	ik = int(0)
	s = np.size(datI)
	datCopy = np.copy(datI)

	period = int(datBase[0,7] * 0.95 * datBase[0,6])
	lim = np.percentile(datI[datI>1000], 10)
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
				ax.plot( datI[a:a+period], linewidth=0.5 )
				ik += 1
			else :
				bolI[0 :s-a-1] = bolI[0 :s-a-1] + datI[a :-1]
				bolC[0 :s-a-1] = bolC[0 :s-a-1]+datC[a :-1]
				borne = np.append( borne, np.array( [a] ) )
				datCopy[a:-1] = 0
				ax.plot( datI[a :a+period], linewidth=0.5 )
				ik += 1

		else :
			datCopy[a] = 0



		bolI = bolI/ik ; bolC = bolC / ik

	return bolI, bolC









######################################
######### Velocity analysis  #########
######################################
def velocityanalysis(mov, c, value) :
	# find the two frame of interest
	img1, img2 = transformation_picture.extractpicture(mov, c, value)

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

