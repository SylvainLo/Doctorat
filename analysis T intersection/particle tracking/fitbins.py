import numpy as np

import matplotlib.pyplot as plt 

import lmfit as lm 

import os 

import fuctionforfit 




####################################
######### fit direct y(x)  #########
####################################

def fitybins(nameDirectory, nFiles, nAllFile, baseValue) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) 
	os.chdir(subDirectoryVideo) ;

	### Lecture data
	datGlobal = np.load('globalAnalysisDataBins.npy') ;

	### Fit of the position y
		# function
	def ydataset(params, position, i) :
		delta = params['delta'].value ;
		xi = params['xi'].value ;
		yIni = params['yIni_{}'.format(i)].value ;
		width = params['width'].value ;
		radius = params['radiusRBC'].value ;
		y0 = params['y0'].value

		x = position ;
		return fuctionforfit.yfit(x, delta, xi, yIni, width, radius, y0) 

		### residual function
	def objective(params, position, data, error) :
		resid = np.array([], dtype=np.float) ; 	
		for i in range(baseValue.analysis_bins-1) :
			parResid = np.array( (data[i,:] - ydataset(params, position[i,:], i)) / error[i,:] ) ;
			resid = np.concatenate((resid, parResid))
		return resid.flatten()

		# Calcul delta, xi et yIni optimal with the method differential_evolution
			# Mise en place parameter
	width = np.mean(datGlobal[0,:,7]) ;
	R0 = 3e-6 #np.mean((np.sqrt(datGlobal[0,8]**2+datGlobal[0,9]**2))/2) ;
	y0 = 0.8e-6 ;

	fitParam0 = lm.Parameters()
	fitParam0.add( 'delta', 0, min=0, max=5)
	fitParam0.add( 'xi', 1e-5, min=1e-10, max=5)
	fitParam0.add( 'width', width, vary=False )
	fitParam0.add( 'radiusRBC', R0, vary=False )
	fitParam0.add( 'y0', y0, vary = False)
	fitParam0.add( 'width', width, vary=False )
	for i in range(baseValue.analysis_bins-1) :
		fitParam0.add( 'yIni_{}'.format(i), datGlobal[i,0,0], vary=False)

			# Calcul fit
	result = lm.minimize(objective, fitParam0, method='differential_evolution', args=(datGlobal[:,:,6],datGlobal[:,:,0], datGlobal[:,:,4]), nan_policy='omit')

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot
	plt.figure('fit bins y(x) for {}'.format(baseValue.name_file))
	for i in range(baseValue.analysis_bins-1) :
		yFromFit = ydataset(result.params, datGlobal[i,:,6], i) ;
		plt.errorbar(datGlobal[i,:,6], datGlobal[i,:,0], yerr=datGlobal[i,:,4], ls='None', marker='o', color=baseValue.color_file[i]) 
		plt.plot(datGlobal[i,:,6], yFromFit, color=baseValue.color_file[i], label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(result.params['delta'].value,result.params['xi'].value,result.params['yIni_{}'.format(i)].value)))
	plt.legend()

	return()





####################################
######### fit y/w=f(x/w)  ##########
####################################

def fitabins(nameDirectory, nFiles, nAllFile, baseValue) :
		### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) 
	os.chdir(subDirectoryVideo) ;

	### Lecture data
	datGlobal = np.load('globalAnalysisDataBins.npy') 

	### Fit of the position a
	def adataset(params, position, i) :
		delta = params['delta'].value ;
		xi = params['xi'].value ;
		aIni = params['aIni_{}'.format(i)].value ;
		R = params['radiusRBC_{}'.format(i)].value ;
		r = params['r_{}'.format(i)].value ;

		b = position ;
		return fuctionforfit.ratiofit(b, delta, xi, aIni, R, r) 

		### residual function
	def objective3(params, position, data, error) :
		resid = np.array([], dtype=np.float) ; 	
		for i in range(baseValue.analysis_bins-1) :
			parResid = np.array( (data[i,:] - adataset(params, position[i,:], i)) / error[i,:] ) ;
			resid = np.concatenate((resid, parResid))
		return resid.flatten()

	### Calcul delta, xi et yIni optimal with the method differential_evolution'
		# Mise en place parameter
	width = np.mean(datGlobal[0,:,7]) ; print(width)
	R0 = 3e-6 
	y0 = 0.8e-6
	bx = datGlobal[:,:,6]/width ;

	fitParam3 = lm.Parameters()
	fitParam3.add( 'delta', 0, min=0, max=10)
	fitParam3.add( 'xi', 0.1, min=1e-20, max=0.1)
	for i in range(baseValue.analysis_bins-1) : 
		fitParam3.add( 'aIni_{}'.format(i), datGlobal[i,0,1], vary=False )
		fitParam3.add( 'radiusRBC_{}'.format(i), R0/width, vary=False )
		fitParam3.add( 'r_{}'.format(i), y0/width, vary=False)

		# Calcul fit
	result = lm.minimize( objective3, fitParam3, method='differential_evolution', args=(bx, datGlobal[:,:,1], datGlobal[:,:,5]), nan_policy='omit' )

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

		### Plot
	plt.figure('fit bins y/w(x/w) for {}'.format(baseValue.name_file))
	for i in range(baseValue.analysis_bins-1) :
		aFromFit = adataset(result.params, datGlobal[i,:,6]/width, i) ;
		plt.errorbar(datGlobal[i,:,6]/width, datGlobal[i,:,1], yerr=datGlobal[i,:,5], ls='None', marker='o', color=baseValue.color_file[i]) 
		plt.plot(datGlobal[i,:,6]/width, aFromFit, color=baseValue.color_file[i], label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(result.params['delta'].value,result.params['xi'].value,result.params['aIni_{}'.format(i)].value)))
	plt.legend()


	return()
