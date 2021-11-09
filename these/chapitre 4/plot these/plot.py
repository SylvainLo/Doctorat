import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import multiprocessing
import glob

import scipy.stats as sci

import functionfit as ff

import basefunction as bf



	### variable experiment
sizePixCam = 5.86 															   # size in um of a pixel of the camera
magnification = 32
pixToMicron = sizePixCam / magnification 										  # coefficient to transform pixel in um




def ploting(dir, h) :
	##################
	# plot
	figC = plt.figure(dir+'Line_concentration', figsize=[5.3, 7])
	figC.subplots_adjust(0.12, 0.08, 0.98, 1, .10, 0)  # (left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

	figD = plt.figure(dir+'mean_CFL', figsize=[5.3, 5.3])
	figD.subplots_adjust( 0.10, 0.1, 0.93, 1, 0.34, 0.14 )

	# directory
	os.chdir(dir)
	pltMov = np.loadtxt('plot.txt')

	(nSubFile, sSubFile) = bf.filestoanalyse( dir, '*%*' )
	for i in range(sSubFile) :
	# sub directory
		# opening directory
		nDirSub = os.path.join(dir, nSubFile[i])
		os.chdir( nDirSub )

		valExp = np.loadtxt('experimental_value.txt', skiprows=1)
		col = ['black', 'red', 'blue', 'cyan']

		# Search and plot data
			# plot 1
		for ii in range(4) :
				# load data
			datI = np.loadtxt('line-intensity_{:02d}.txt'.format(int(pltMov[i,ii])))
			posLine = np.loadtxt( 'yLine_{:02d}.txt'.format(int(pltMov[i,ii])), dtype=np.int)
			posWallT = np.loadtxt( 'wall-top_{:02d}.txt'.format(int(pltMov[i,ii])), dtype=np.int )
			posWallB = np.loadtxt( 'wall-bot_{:02d}.txt'.format(int(pltMov[i,ii])), dtype=np.int )

				# value x axis and y axis
			w = (posWallB[posLine]-posWallT[posLine])*pixToMicron
			y = np.linspace( 0, w, np.size(datI) )
			intensity = datI/np.max(datI)
			x = valExp[int(pltMov[i,ii]-1),1] + posLine*pixToMicron

				# plot
			k = int(4*i)+ii+1
			ax = figC.add_subplot(4,4,k)
			ax.plot(y, np.flipud(intensity), color=col[i])
			ax.set_xlim( 0, y.max() ) ; ax.set_ylim( 0, 1.15 )

			ax.text( y[5], 1.02, r'$x$ = {:.0f} $\mu $m'.format( x ))#, horizontalalignment='right', verticalalignment='top' )

			if k == 1 or k == 5 or k == 9 or k==13:
					ax.set_ylabel( r'$\Phi//\Phi_0$' )
			else :
					plt.setp( ax.get_yticklabels( ), visible=False )

			if k > 12 :
					ax.set_xlabel( r'$y \mu m $' )
			else :
					plt.setp( ax.get_xticklabels( ), visible=False )

			# Search and plot data
				# plot 1
		nMovie = glob.glob( '*.avi' )
		sMovie = len( nMovie )

				# variable for plot
		yMean = np.empty(sMovie)
		wMean = np.empty(sMovie)
		yMedian = np.empty(sMovie)
		yStd = np.empty(sMovie)
		intSkew = np.empty(sMovie)
		y80 = np.empty(sMovie)
		y20 = np.empty(sMovie)
		x = np.empty(sMovie)

		for ii in range(sMovie) :
					# load data
			cfl = np.loadtxt( 'cfl_along_x.txt' ) * pixToMicron
			datI = np.loadtxt( 'line-intensity_{:02d}.txt'.format(ii+1) )
			posLine = np.loadtxt( 'yLine_{:02d}.txt'.format(ii+1), dtype=np.int )
			posWallT = np.loadtxt( 'wall-top_{:02d}.txt'.format(ii+1), dtype=np.int )
			posWallB = np.loadtxt( 'wall-bot_{:02d}.txt'.format(ii+1), dtype=np.int )

					# value x axis and y axis
			w = (posWallB[posLine]-posWallT[posLine])*pixToMicron
			wMean[ii] = w
			y = np.linspace( 0, w, np.size(datI) )
			pdf = np.cumsum(datI) / np.sum(datI)

					# fill vector
			x[ii] = valExp[ii, 1] + posLine*pixToMicron
			yMean[ii] = np.sum(y*datI) / np.sum(datI)
			yMedian[ii] = y[int(np.argmax(pdf>0.49))]
			y20[ii] = y[int(np.argmax(pdf>0.195))] ; y80[ii] = y[int(np.argmax(pdf>0.795))]
			yStd[ii] = np.sqrt(np.sum(datI * (y-yMean[ii])**2) / np.sum(datI))

			intSkew[ii] = np.sum(datI * (y-yMean[ii])**3) / (np.sum(datI) * yStd[ii]**3)

		plt.figure('test_{}'.format(dir))
		plt.plot(x, y80-y20)
		plt.figure('test_test_{}'.format(dir))
		plt.plot(x, yMedian-y20 )
		plt.plot(x, y80-yMedian, '--')
		plt.plot(x, cfl[1,:], ':')

		colorR = 'tab:red' ; colorB = 'tab:blue'
		ax = figD.add_subplot(2,2,i+1)
		ax.set_ylim( 0, np.max(wMean) )
		ax.tick_params( axis='y', labelcolor=colorR )

		axtwin = ax.twinx( )
		axtwin.tick_params( axis='y', labelcolor=colorB )
		if np.max(np.abs(intSkew)) < 1 :
			axtwin.yaxis.set_major_locator( ticker.MultipleLocator( 0.1 ) )
			print('OK')

		axtwin.plot( x/1000, np.abs( intSkew ), '-.', color=colorB )

		ax.plot(x/1000, wMean-yMedian, 'o', color=colorR)
		ax.plot(x/1000, wMean-cfl[0,:], ':', color=colorR)
		ax.plot(x/1000, cfl[1,:], ':', color=colorR)
		ax.fill_between(x/1000, w-y20, w-y80, alpha=0.25, facecolor=colorR)

		axtwin.yaxis.set_major_formatter( FormatStrFormatter( '%.1f' ) )
		if i == 0 or i == 2 :
			ax.set_ylabel( r'$y$ ($\mu$m)', color=colorR )
		if i > 1 :
			ax.set_xlabel( r'$x$ (mm)' )







		ff.fitymean( x, w-yMedian, np.mean(wMean)/2, h, 2.6, ax )


	return figD, figC

