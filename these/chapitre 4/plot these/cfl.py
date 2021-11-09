import numpy as np
import cv2
import math
import scipy.signal as scisig
import matplotlib.pyplot as plt

import transformation_picture





def functioncfl(movie, back, edt, edb, y) :
	"""
	:param movie: movie object cv2
	:param back : background of the movie
	:param edt : point in the wall top
	:param edb : point in the wall bot
	detection size cell free layer or cfl
	"""

	### Constant
		# movie contstant
	movie.set(cv2.CAP_PROP_POS_FRAMES, 0)           													  # base index 0
	tF = int( movie.get(cv2.CAP_PROP_FRAME_COUNT))
	wi = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))

	x = np.linspace( 0, wi-1 , wi, dtype=np.int )


		# search threshold automatically (using first picture)
	movie.set( 1, 0 )
	k = 0 ; ret = True
	while k==0 and ret == True :
		ret, img = movie.read()
		img = img[:, :, 0]
		if np.max(back - img) > 50 :
			k=1
			valueThreshold = searchthreshold(img, back, x, edt, edb)

		# counter frame
	t = 0

		# evolution CFL over time
	deltaT = np.empty(tF)
	deltaB = np.empty(tF)


	### Read Movie
	movie.set(1, 0)
	while movie.isOpened():
		# Opening one picture of the movie
		ret, frame = movie.read()
		t += 1
		if t%200 == 0 :
			print(t)

		# index of the frame
		nF = int( movie.get(cv2.CAP_PROP_POS_FRAMES) - 1)
		if ret is False :  																	# if no more frame then stop
			break

		# modification of the frame (crop/subtract background/median filtering)
		frame = frame[:, :, 0] 		#;    cv2.imshow('normal', frame) ; cv2.waitKey(1)
		frame2 = cv2.subtract(back, frame)			;    cv2.imshow('subtract',frame2[:,y-20:y+21]) ; cv2.waitKey(1)

		# subtract background
			# personnal method
		ret, threshPic = cv2.threshold(frame2, valueThreshold, 255, cv2.THRESH_BINARY) ; cv2.imshow('threshold', threshPic[:,y-20:y+21]) ; cv2.waitKey(1)

			# combination of the two method
		picMed = cv2.medianBlur(threshPic, 3)		             #;  cv2.imshow('median filtering',picMed) ; cv2.waitKey(1)

		# measure of the CFL
		delT = 0
		delB = 0
		ii = int(0)

		for i in range(y-20,y+21,1) :
			wT = edt[i] ; wB = edb[i]

			# particular case no RBC between walls then CFL = size between wall divided by 2
			if np.sum(picMed[wT:wB,i] ) > 3 :

				# measure CFL bot
				pB = 1
				while picMed[wB-pB, i] < 200 and picMed[wB-pB-1, i] < 200 :
					pB += 1
				delB = delB + pB - 1

				# search the cfl top
				pT = 1
				while picMed[wT+pT, i] < 200 :
					pT += 1
				delT = delT + pT - 1

				ii += 1


		# CFL over time
		if ii is 0 :
			deltaT[nF] = np.nan
			deltaB[nF] = np.nan

		else :
			deltaT[nF] = delT / ii
			deltaB[nF] = delB / ii

	return deltaT, deltaB




def searchthreshold(img, b, xx, top, bot) :
	# subtract first picture with background
	picDiff = cv2.subtract(b, img)

	### Threshold
		# Create windows
	cv2.namedWindow( 'threshold', cv2.WINDOW_AUTOSIZE )
	cv2.createTrackbar( 'Threshold', 'threshold', 0, 255, nothing )
	cv2.createTrackbar( 'Kernel', 'threshold', 1, 20, nothing)
	cv2.createTrackbar( '0 : OFF \n1 : ON', 'threshold', 0, 1, nothing )


		# Threshold by user
	while 1 :
		k = cv2.getTrackbarPos( '0 : OFF \n1 : ON', 'threshold' )
		if k == 1 :
			break
		thrVal = cv2.getTrackbarPos( 'Threshold', 'threshold' )
		k = cv2.getTrackbarPos( 'Kernel', 'threshold' )

			# threshold
		ret, picThr = cv2.threshold( picDiff, thrVal, 255, cv2.THRESH_BINARY )
		kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (k+1, k+1) )
		close = cv2.morphologyEx(picThr, cv2.MORPH_OPEN, kernel)

			# plot
		picThr[picThr>100] = 1
		result = picThr*img
		result[top, xx] = 255 ; result[bot, xx] = 255
		cv2.imshow( 'threshold',  result) ; cv2.waitKey(1)
		cv2.imshow( 'closing', close ) ; cv2.waitKey( 1 )

	# Close windows
	cv2.destroyAllWindows()

	return thrVal

def nothing(x) :
	pass


