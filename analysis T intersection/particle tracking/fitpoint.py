import numpy as np

import matplotlib.pyplot as plt

import lmfit as lm

import os

import fuctionforfit






####################################
######### fit direct y(x)  #########
####################################
def fitypoint(nameDirectory, nFiles, nAllFile, baseValue) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) 
	os.chdir(subDirectoryVideo)

	### Lecture data
	datPoint = np.load('totalLocalAnalysisData.npy') 

		# Reorganize data
	vectorSort = np.argsort(datPoint[:,1])
	xNoOrder = datPoint[:,1] ; xOrder = xNoOrder[vectorSort]
	dNoOrder = datPoint[:,3] ; yOrder = dNoOrder[vectorSort]

	### Fit of the position y
	def ydataset(params, position) :
		delta = params['delta'].value
		xi = params['xi'].value
		yIni = params['yIni'].value
		width = params['width'].value
		radius = params['radiusRBC'].value

		x = position
		return fuctionforfit.yfitsimple(x, delta, xi, yIni, width, radius)

	### residual function
	def objective(params, position, data, error) :
		residual = (data - ydataset(params, position))/error
		return residual.flatten()

	### Calcul delta, xi et yIni optimal with the method differential_evolution
		# Mise en place parameter
	width = np.mean(datPoint[:,5])
	R0 = 2.8e-6
	error = 1e-6

	fitParam0 = lm.Parameters()
	fitParam0.add( 'delta', 2, min=0, max=10)
	fitParam0.add( 'xi', 1e-5, min=1e-10, max=5)
	fitParam0.add( 'yIni' , yOrder[0], min=0, max=np.mean(yOrder) )
	fitParam0.add( 'width', width, vary=False )
	fitParam0.add( 'radiusRBC', R0, vary=False )

		# Calcul fit
	result = lm.minimize(objective, fitParam0, method='Nelder', args=(xOrder, yOrder, error), nan_policy='omit')

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot
	plt.figure('fit point y(x) for {}'.format(baseValue.name_file))
	yFromFit = ydataset(result.params, xOrder)
	plt.plot(xOrder, yOrder, marker='o', linestyle='None', markersize=1, alpha=0.2)
	plt.plot(xOrder, yFromFit, 'r-', label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.
										 format(result.params['delta'].value,result.params['xi'].value,result.params['yIni'].value)))
	plt.legend()

	return()







####################################
######### fit y/w=f(x/w)  ##########
####################################
def fitapoint(nameDirectory, nFiles, nAllFile, baseValue) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) 
	os.chdir(subDirectoryVideo)

	### Lecture data
	datPoint = np.load('totalLocalAnalysisData.npy') 

		# Reorganize data
	vectorSort = np.argsort(datPoint[:,1])
	xNoOrder = datPoint[:,1] ; xOrder = xNoOrder[vectorSort]
	dNoOrder = datPoint[:,6] ; a = dNoOrder[vectorSort]

	### Fit of the position a
	def adataset(params, position) :
		delta = params['delta'].value
		xi = params['xi'].value
		aIni = params['aIni'].value
		R = params['radiusRBC'].value

		b = position
		return fuctionforfit.ratiofitsimple(b, delta, xi, aIni, R)

		### residual function
	def objective3(params, position, data, error) :
		residual = (data - adataset(params, position))/error
		return residual.flatten()

	### Calcul delta, xi et yIni optimal with the method differential_evolution'
		# Mise en place parameter
	width = np.mean(datPoint[:,5])
	R0 = 2.84e-6
	bx = xOrder/width
	error = 1e-6/width

	fitParam3 = lm.Parameters()
	fitParam3.add( 'delta', 0, min=0, max=10)
	fitParam3.add( 'xi', 0.1, min=1e-20, max=0.1)
	fitParam3.add( 'aIni', a[0], min=0, max=np.mean(a))
	fitParam3.add( 'radiusRBC', R0/width, vary=False )

		# Calcul fit
	result = lm.minimize( objective3, fitParam3, method='Nelder', args=(bx, a, error), nan_policy='omit' )

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot fit
	plt.figure('fit y/w(x/w) for {}'.format(baseValue.name_file))
	aFromFit = adataset(result.params, bx)
	plt.plot(bx, a, color='b', marker='o', ls='None', markersize=1, alpha=0.2) 
	plt.plot( bx, aFromFit, ls='-', color='r', label = ('fit : delta={:.2E}, chi={:.2E}, aIni={:.2E}'
														.format(result.params['delta'].value, result.params['xi'].value, result.params['aIni'].value)) )
	plt.legend()

	return

