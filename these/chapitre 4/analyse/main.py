import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing
import glob
import cv2
import math
import time


import transformation_picture
import basefunction as bf
import studymovie
import cfl


####################################################################
### Global variable ###
	### print commentary
bf.introduction()
time.sleep(1)


	### choose on which data you want to work
backAnswer = bf.question('Do you want to create new backgrounds ? (1(yes)/0(no)) \n')
if backAnswer is '1' :
	bf.backgroundrule() ; time.sleep(1)

wallAnswer = bf.question('Do you want to detect the position of the wall ? (1(yes)/0(no)) \n')
if wallAnswer is '1' :
	bf.wallrule() ; time.sleep(1)

lineAnswer = bf.question('Do you want to choose new line ? (1(yes)/0(no)) \n')
if lineAnswer is '1' :
	bf.linerule() ; time.sleep(1)

lineIntAnswer = bf.question('Do you want to measure the new intensity on a profile line ? (1(yes)/0(no)) \n')

cflAnswer = bf.question('Do you want to measure the size of the cfl ? (1(yes)/0(no)) \n')
if cflAnswer is '1':
	bf.cflrule() ; time.sleep(1)

####################################################################
### Main code ###
if __name__ == '__main__' :
	##################
	### Main directory (sort of root directory)
		# main directory
	nDir = 'J:/evolution_profile_hemato/'
	nDir = bf.control(nDir)

		# search folder in this directory containing "*IMAGE*"
	os.chdir( nDir ) ; print('Your directory is : ', nDir)
	(nameFile, sFile) = bf.filestoanalyse( nDir, '*IMAGE*' )


	for n in nameFile :
		# search in each folder "*IMAGE*", the folder "*%*"
		nDirSub = os.path.join( nDir, n )
		os.chdir( nDirSub )
		(nSubFile, sSubFile) = bf.filestoanalyse( nDirSub, '*%*' )


		for nn in nSubFile :
		# open folder "*%*" to extract data
			# opening directory
			nDirSub2 = os.path.join( nDirSub, nn )
			os.chdir( nDirSub2 )

			# creation plot
				# plot profile Intensity
			fig1 = plt.figure( 'intensity line ' + n + ' ' + nn )

				# plot profile Intensity all channel in one graph
			fig11 = plt.figure( 'intensity line over channel ' + n + ' ' + nn )
			ax11 = fig11.add_subplot( 111 )

				# plot cell free layer
			fig2 = plt.figure( 'cell free layer ' + n + ' ' + nn )
			ax2 = fig2.add_subplot( 111 )


			#################
			# Opening of the data
				# detect all movies
			nMovie = glob.glob('*.avi')
			sMovie = len(nMovie)   # number of movies

				# Variable
					# variable so that data correspond
			expVal = np.loadtxt('experimental_value.txt', skiprows=1)
			expVal = expVal[:sMovie]

					# creation variable cfl over x
			cflXTop = np.empty(sMovie)
			cflXBot = np.empty(sMovie)

					# width channel
			if os.path.isfile('width.txt') :
				w = np.loadtxt('width.txt')
			else :
				w = np.ones(sMovie)

					# for plot
			linePlot = math.ceil(sMovie/3)


			#################
			# opening of the movie
			for m in range(sMovie):
				mov = cv2.VideoCapture(nMovie[m])  # object movie in Python (doesn't charge all the movie on the RAM)
				fps = int(mov.get(cv2.CAP_PROP_FPS))  												  # frame per second
				print('\n', '		',nMovie[m])


			#################
				# Background, mean picture and minimum picture
				if os.path.isfile('background_%02d.bmp'%(m+1)):
					if backAnswer is '1' :
						print('				The background will be measured before going to the next step')
						background, meanInt, maxInt = transformation_picture.backgroundcreation(mov)
						cv2.imwrite( 'background_%02d.bmp'%(m+1), background )
						cv2.imwrite( 'mean-intensity_%02d.bmp' % (m+1), meanInt )
						cv2.imwrite( 'max-intensity_%02d.bmp'%(m+1), maxInt )
					else :
						print('				The background is already measure, we will go to the next step')
						background = cv2.imread('background_%02d.bmp'%(m+1), cv2.IMREAD_GRAYSCALE)
						meanInt = cv2.imread( 'mean-intensity_%02d.bmp' % (m+1), cv2.IMREAD_GRAYSCALE)
						maxInt = cv2.imread( 'max-intensity_%02d.bmp'%(m+1), cv2.IMREAD_GRAYSCALE )
				else :
					print('				The background need to be measured before going to the next step')
					background, meanInt, maxInt = transformation_picture.backgroundcreation(mov)
					cv2.imwrite( 'background_%02d.bmp'%(m+1), background )
					cv2.imwrite( 'mean-intensity_%02d.bmp' % (m+1), meanInt )
					cv2.imwrite( 'max-intensity_%02d.bmp'%(m+1), maxInt )


				# Invert Picture
				backgroundIv = cv2.bitwise_not(background)
				meanIntIv = cv2.bitwise_not(meanInt)
				maxIntInv = cv2.bitwise_not(maxInt)


			#################
				# extract position wall
				if os.path.isfile('wall-top_%02d.txt'%(m+1)) and os.path.isfile('wall-bot_%02d.txt'%(m+1)):
					if wallAnswer is '1' :
						print('				The position of the wall will be measured before going to the next step')
						edgeTop, edgeBot = transformation_picture.detectionedges( meanInt )
						np.savetxt('wall-top_%02d.txt'%(m+1), edgeTop, delimiter='	')
						np.savetxt('wall-bot_%02d.txt'%(m+1), edgeBot, delimiter='	')
						w[m] = np.abs( np.mean( np.subtract(edgeBot, edgeTop) ) ) *  expVal[m, 5] / expVal[m, 4]

					else :
						print('				Edges already detected')
						edgeTop = np.loadtxt('wall-top_%02d.txt'%(m+1), dtype=np.int)
						edgeBot = np.loadtxt('wall-bot_%02d.txt'%(m+1), dtype=np.int )
				else :
					print('				The position of the wall need to be measured before going to the next step')
					edgeTop, edgeBot = transformation_picture.detectionedges( meanInt )
					np.savetxt( 'wall-top_%02d.txt'%(m+1), edgeTop, delimiter='	' )
					np.savetxt( 'wall-bot_%02d.txt'%(m+1), edgeBot, delimiter='	' )
					w[m] = np.abs( np.mean( np.subtract(edgeBot, edgeTop) ) ) * expVal[m, 5] / expVal[m, 4]



			#################
			### Intensity on a line
				# choose line
				if os.path.isfile( 'yLine_%02d.txt'%(m+1)) :
					if lineAnswer is '1' :
						print('				The line on which the hematocrit profil is measured will be chosen')
						posYLine = studymovie.selectionline(meanIntIv)
						saveLine = open('yLine_%02d.txt'%(m+1),'w')
						saveLine.write( str(posYLine) )
						saveLine.close()

					else :
						print('				We already choose a line')
						saveLine = open( 'yLine_%02d.txt'%(m+1), 'r' )
						posYLine = int(saveLine.read())
						saveLine.close( )

				else :
					print('				The line on which the hematocrit profil is measured will be chosen')
					posYLine = studymovie.selectionline(meanIntIv)
					saveLine = open( 'yLine_%02d.txt'%(m+1), 'w' )
					saveLine.write( str(posYLine) )
					saveLine.close()


				# measure intensity along the line
				ax1 = fig1.add_subplot(int(linePlot), 3, int(m+1))

				if os.path.isfile( 'line-intensity_%02d.txt'%(m+1)) :
					if lineIntAnswer is '1' or lineAnswer is '1':
						logInt = transformation_picture.lineintensity( meanIntIv, backgroundIv, posYLine, edgeTop, edgeBot )
						np.savetxt( 'line-intensity_%02d.txt'%(m+1), logInt )
					else :
						logInt = np.loadtxt( 'line-intensity_%02d.txt'%(m+1) )
				else :
					logInt = transformation_picture.lineintensity( meanIntIv, backgroundIv, posYLine, edgeTop, edgeBot )
					np.savetxt( 'line-intensity_%02d.txt'%( m+1 ), logInt )


				ax1.plot( logInt, label='movie:{}'.format( m+1 ) )
				ax1.set_ylim(bottom=0)
				ax11.plot( logInt, label='movie:{}'.format( m+1 ) )


			#################
			### Search of the CFL
				# CFL over time
				if os.path.isfile( 'cfl-Top_%02d.txt'%(m+1)) and os.path.isfile( 'cfl-Bot_%02d.txt'%(m+1)) :
					if cflAnswer is '1' :
						print('				We will measure the cfl')
						cflTop, cflBot = cfl.functioncfl( mov, background, edgeTop, edgeBot, posYLine )
						np.savetxt('cfl-Top_%02d.txt'%(m+1), cflTop, delimiter='	')
						np.savetxt('cfl-Bot_%02d.txt'%(m+1), cflBot, delimiter='	')
					else :
						print('				We already measure the cfl')
						cflTop = np.loadtxt('cfl-Top_%02d.txt'%(m+1))
						cflBot = np.loadtxt('cfl-Bot_%02d.txt'%(m+1))

				else :
					print('				The cfl will be measured')
					cflTop, cflBot = cfl.functioncfl( mov, background, edgeTop, edgeBot, posYLine )
					np.savetxt( 'cfl-Top_%02d.txt'%(m+1), cflTop, delimiter='	' )
					np.savetxt( 'cfl-Bot_%02d.txt'%(m+1), cflBot, delimiter='	' )

				# CFL over x
				cflXTop[m] = np.nanmean(cflTop)
				cflXBot[m] = np.nanmean(cflBot)


			ax2.plot(expVal[:,1], cflXTop, 'r', label='cfl top')
			ax2.plot(expVal[:,1], cflXBot, 'b', label='cfl bot')
			np.savetxt('cfl_along_x.txt', np.array([cflXTop, cflXBot]), delimiter='	')



	plt.show()