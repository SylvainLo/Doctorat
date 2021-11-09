import numpy as np
from math import cosh, pi

import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import matplotlib.ticker as plttick
import corner

import lmfit as lm

import os

import fuctionforfit



##########################
######### residu #########
##########################
def meshgridbrutemethod(nameDirectory) :
	### creation delta, xi
	nDelta = 61 ; delta = np.linspace(-1,5,nDelta)
	xi = np.array([ np.linspace(5e-4,5e-1,1000,endpoint=False) ]) ; xi = np.reshape(xi,np.size(xi))

	### Creation of the grid
	X,Y = np.meshgrid(delta, xi, indexing='ij', sparse=False)
	
	### Change of directory
	os.chdir(nameDirectory)
	np.save('Xforresidual', X) ; np.save('Yforresidual', Y)

	return X,Y



###############################################################################
def brutemethod(nameDirectory, nFiles, nAllFile, baseValue, X, Y) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) ; print(subDirectoryVideo)
	os.chdir(subDirectoryVideo)


	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt')


	### Brute method 
		# Grid results
	(nDelta,nXi) = np.shape(X)
	residual = np.empty((nDelta,nXi))

		# Constant value
	width = np.mean(datGlobal[:,1])
	R0 = 2.84e-6
	height = baseValue.distance[0,9]
	A = cosh(pi * width / height)
	B = cosh(3 * pi * width / height)

	for i in range(nDelta) :
		for j in range(nXi) :
			residual[i,j] = np.sum( np.sqrt( (( datGlobal[:,2] - fuctionforfit.yfit(
				datGlobal[:,0], X[i,0], Y[0,j], 0, datGlobal[0,2], width, R0, height, A, B) ) / datGlobal[:,3])**2 ) )


	### Save data
	residualFile = 'matriceResidual'
	np.save(residualFile, abs(residual))
	print('residual for {}'.format(subDirectoryVideo))

	return









###############################
######### Plot residu #########
###############################
def searchgrid(nameDirectory) :
	### Change directory
	os.chdir(nameDirectory)

	### Take back the grid
	X = np.load('Xforresidual.npy') ; Y =np.load('Yforresidual.npy')

	return X,Y



###############################################################################
def plotresidu(nameDirectory, nFiles, nAllFile, baseValue, ax, X, Y) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) ; print(subDirectoryVideo)
	os.chdir(subDirectoryVideo)


	### Lecture data
	residual = np.load('matriceResidual.npy')
	(line, column) = np.shape(residual)


	### Modification data
		# Mask residu
	limResidual = np.percentile(residual,1)
	boolResidual = np.zeros((line, column), dtype=int)

		# Threshold residu
	boolResidual[residual < limResidual] = 1
	(xMin, yMin) = np.unravel_index(np.argmin(residual, axis=None), residual.shape)
	print('the residual = ',residual[xMin, yMin], 'delta = ', X[xMin,0], 'xi = ', Y[0,yMin], 'xMin = ', xMin, 'yMin = ', yMin)


	### Plot
		# Individual plot
			# New colormap
	cMap = pltcolor.ListedColormap([(1,1,1,0), pltcolor.to_rgba(baseValue.color_file[nFiles],0.2)])

			#Plot
				# Plot 1
	fig1 = plt.figure('plot residual for {}'.format(baseValue.name_file))
	ax11 = fig1.add_axes([0.15,0.1,0.8,0.85])
	ax11.set_xlabel(r'$\delta$') ; ax11.set_ylabel(r'$\xi$')
	ax11.contourf(X, Y, boolResidual, cmap=cMap)
	ax11.set_yscale('log')

				# Plot 2
	fig2 = plt.figure('plot residual precisely for {}'.format(baseValue.name_file))
	ax21 = fig2.add_axes([0.15,0.1,0.8,0.85])
	ax21.set_xlabel(r'$\delta$') ; ax21.set_ylabel(r'$\xi$')
	level = plttick.MaxNLocator(nbins=100).tick_values(residual.min(), residual.max())
	ax21.contourf(X, Y, residual, levels=level, cmap='viridis')
	ax21.set_yscale('log')

			# Save plot
	fig1.savefig('residual.pdf', frameon=False, bbox_inch='tight', orientation='landscape')
	fig2.savefig('precise residual.pdf', frameon=False, bbox_inch='tight', orientation='landscape')

		# Collective plot
	ax.contourf(X, Y, boolResidual, cmap=cMap)

	return ax







###############################
######### Sum residu ##########
###############################
def sumresidu(nameDirectory, nAllFile, sizeAllFile) :
	### Creation grid
		# grid
	X,Y = searchgrid(nameDirectory)
	n = np.shape(X)

		# creation vector data
	vectorSum = np.zeros( (n[0], n[1]) )


	### Sum all residual
	for nFiles in range(sizeAllFile) :
		# Change of directory
		subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) ; print(subDirectoryVideo)
		os.chdir(subDirectoryVideo)

		# Lecture data
		residual = np.load('matriceResidual.npy')

		# Modification data
		vectorSum = vectorSum + residual


	### Minimum
	(xC, yC) = np.unravel_index(np.argmin(vectorSum, axis=None), (n[0], n[1]))


	### Plot
	fig = plt.figure('plot sum all residual')
	ax1 = fig.add_axes([0.15,0.1,0.8,0.85])
	cMapVeridis = plt.cm.get_cmap('viridis')
	ax1.set_xlabel(r'$\delta$') ; ax1.set_ylabel(r'$\xi$')
	level = plttick.MaxNLocator(nbins=100).tick_values(vectorSum.min(), vectorSum.max())
	plot = ax1.contourf(X, Y, vectorSum, levels=level, cmap=cMapVeridis)
	plt.plot(X[xC,0], Y[0, yC], linestyle='None', marker='o', color='r', markersize=5)
	ax1.set_yscale('log')
	fig.colorbar(plot, ax=ax1)

			# Save plot
	os.chdir(nameDirectory)
	fig.savefig('sumresidual.pdf', frameon=False, bbox_inch='tight', orientation='landscape')
	np.savetxt('value residual.txt', np.array([X[xC,0], Y[0, yC]]))

	return








##############################################
######### Fit by Monte Carlo method  #########
##############################################
def montecarlofity(nameDirectory, nFiles, nAllFile, baseValue) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) 
	os.chdir(subDirectoryVideo)


	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt') 


	### Basic fit with Nelder
		# function
	def ydataset(params, position) :
		delta = params['delta'].value
		xi = params['xi'].value
		yIni = params['yIni'].value
		width = params['width'].value
		radius = params['radiusRBC'].value

		x = position
		return fuctionforfit.yfitsimple(x, delta, xi, yIni, width, radius)

		# residual function
	def objective(params, position, data, error) :
		residual = (data - ydataset(params, position))/error
		return residual.flatten()

		# Calcul delta, xi et yIni optimal with the method differential_evolution
			# Mise en place parameter
	width = np.mean(datGlobal[:,1])
	R0 = 2.84e-6
	yInitial = datGlobal[0,2] ; yInitialError = datGlobal[0,3]

	fitParam = lm.Parameters()
	fitParam.add( 'delta', 0, min=0, max=10 )
	fitParam.add( 'xi', 1e-5, min=1e-10, max=1 )
	fitParam.add( 'yIni' , datGlobal[0,2], min=yInitial-yInitialError, max=yInitial+yInitialError )
	fitParam.add( 'width', width, vary=False )
	fitParam.add( 'radiusRBC', R0, vary=False )

			# Calcul fit
	result = lm.minimize(objective, fitParam, method='Nelder', args=(datGlobal[:,0], datGlobal[:,2], datGlobal[:,3]), nan_policy='omit')


	### Log-posterior probability
		# Add a noise parameter
	result.params.add('noise', value=1, min=1e-8, max=100)

		# This is the log-likelihood probability for the sampling
	def lnprob(params, position, data, error) :
		noise = params['noise'].value
		residual = (data - ydataset(params, position)) / error
		logProb = -0.5 * np.sum( (residual/noise)**2 + np.log(2* np.pi* noise**2) )
		return logProb

		# Set Minimizer
	resultMini = lm.Minimizer( lnprob, result.params, fcn_args=(datGlobal[:,0], datGlobal[:,2], datGlobal[:,3]) )
	res = resultMini.emcee(burn=300, steps=10000, thin=20, params=result.params)
	print('----------------{}----------------'.format(nAllFile[nFiles]))
	lm.report_fit(res.params)


		# Plot results
	corner.corner( res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()),
				   title_kwargs='monte carlo for {}'.format(baseValue.name_file) )
	plt.savefig('monte carlo.pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape')

	return




###############################################################################
def montecarlofitytN(nameDirectory, nFiles, nAllFile, baseValue) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)


	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt')


	### Basic fit with Nelder
		# function
	def ydataset(params, position) :
		lamb = params['lamb'].value
		xi = params['xi'].value
		yIni = params['yIni'].value

		t = position
		return fuctionforfit.fitdirect(t, lamb, xi, yIni)

		# residual function
	def objective(params, position, data, error) :
		residual = (data - ydataset(params, position))/error
		return residual.flatten()

		# Calcul delta, xi et yIni optimal with the method differential_evolution
			# Mise en place parameter
	R0 = 2.84e-6
	y = datGlobal[:,2] / R0
	yError = datGlobal[:,3] / R0

	fitParam = lm.Parameters()
	fitParam.add( 'lamb', 2, min=0, max=10)
	fitParam.add( 'xi', 1e-5, min=1e-10, max=1)
	fitParam.add( 'yIni' , y[0], min=y[0]-yError[0], max=y[0]+yError[0] )

			# Calcul fit
	result = lm.minimize(objective, fitParam, method='Nelder', args=(datGlobal[:,17], y, yError), nan_policy='omit')


	### Log-posterior probability
		# Add a noise parameter
	result.params.add('noise', value=1, min=1e-8, max=100)

		# This is the log-likelihood probability for the sampling
	def lnprob(params, position, data, error) :
		noise = params['noise'].value
		residual = (data - ydataset(params, position)) / error
		logProb = -0.5 * np.sum( (residual/noise)**2 + np.log(2* np.pi* noise**2) )
		return logProb

		# Set Minimizer
	resultMini = lm.Minimizer( lnprob, result.params, fcn_args=(datGlobal[:,17], y, yError) )
	res = resultMini.emcee(burn=300, steps=5000, thin=20, params=result.params)
	print('----------------{}----------------'.format(nAllFile[nFiles]))
	lm.report_fit(res.params)

		# Show result :
	quantiles = np.percentile(res.flatchain['xi'], [2.28, 15.9, 50, 84.2, 97.7])
	print( '2 sigma = ', 0.5*(quantiles[4]-quantiles[0]) )

		# Plot results
	corner.corner( res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()),
				   title_kwargs='monte carlo for {}'.format(baseValue.name_file) )
	plt.savefig('monte carlo of y(tN).pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape')

	return