import numpy as np

import matplotlib.pyplot as plt

import lmfit as lm

import os

import fuctionforfit


####################################
######### fit direct y(x)  #########
####################################

def fityxglobal(nameDirectory, nFiles, nAllFile, baseValue) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) 
	os.chdir(subDirectoryVideo)

	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt') 

###############################################################################
	### Fit of the position y
		# function
	def ydataset(params, position) :
		delta = params['delta'].value
		y0 = params['y0'].value
		xi = params['xi'].value
		yIni = params['yIni'].value
		width = params['width'].value
		radius = params['radiusRBC'].value

		x = position
		return fuctionforfit.yfit(x, delta, xi, yIni, width, radius, y0)

		### residual function
	def objective(params, position, data, error) :
		residual = (data - ydataset(params, position))/error
		return residual.flatten()

		# Calcul delta, xi et yIni optimal with the method differential_evolution
			# Mise en place parameter
	width = np.mean(datGlobal[:,1])
	R0 = 2.84e-6
	yIni = datGlobal[0,2] ; yIniLim = datGlobal[0,3]
	y0 = 0

	fitParam0 = lm.Parameters()
	fitParam0.add( 'delta', 0, min=0, max=5)
	fitParam0.add( 'xi', 0, min=1e-10, max=1)
	fitParam0.add( 'yIni' , yIni, min=yIni-yIniLim, max=yIni+yIniLim )
	fitParam0.add( 'y0', y0, vary=False )
	fitParam0.add( 'width', width, vary=False )
	fitParam0.add( 'radiusRBC', R0, vary=False )

			# Calcul fit
	result = lm.minimize(objective, fitParam0, method='Nelder', args=(datGlobal[:,0],datGlobal[:,2], datGlobal[:,3]), nan_policy='omit')

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot
	plt.figure('fit delta and xi for y(x) ({})'.format(baseValue.name_file))
	plt.xlabel('x (m)') ; plt.ylabel('y (m)')
	yFromFit = ydataset(result.params, datGlobal[:,0])
	print('residu = ', np.sum( np.sqrt((datGlobal[:,2] - yFromFit)/datGlobal[:,3])**2 ) )
	plt.errorbar(datGlobal[:,0], datGlobal[:,2], yerr=datGlobal[:,3], fmt='bo')
	plt.plot(datGlobal[:,0], yFromFit, 'r-', label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'
													  .format(result.params['delta'].value,result.params['xi'].value,result.params['yIni'].value)))
	plt.legend()






####################################
######### fit y/w=f(x/w)  ##########
####################################
def fitaxglobal(nameDirectory, nFiles, nAllFile, baseValue) :
		### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) 
	os.chdir(subDirectoryVideo)

	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt') 

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
		resid = (data - adataset(params, position))/error
		return resid.flatten()

	### Calcul delta, xi et yIni optimal with the method differential_evolution'
		# Mise en place parameter
	width = np.mean(datGlobal[0,1])
	R0 = 2.84e-6
	bx = datGlobal[:,0]/datGlobal[:,1]
	aIni = datGlobal[0,4] ; aIniLim = datGlobal[0,5]

	fitParam3 = lm.Parameters()
	fitParam3.add( 'delta', 0, min=0, max=10)
	fitParam3.add( 'xi', 0.1, min=1e-20, max=0.1)
	fitParam3.add( 'aIni', aIni, min=aIni-aIniLim, max=aIni+aIniLim )
	fitParam3.add( 'radiusRBC', R0/width, vary=False )

		# Calcul fit
	result = lm.minimize( objective3, fitParam3, method='Nelder', args=(bx, datGlobal[:,4], datGlobal[:,5]), nan_policy='omit' )

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot fit
	plt.figure('fit delta and xi for y/w(x/w) ({})'.format(baseValue.name_file))
	plt.xlabel('x/w') ; plt.ylabel('y/w')
	aFromFit = adataset(result.params, bx)
	plt.errorbar(bx, datGlobal[:,4], yerr=datGlobal[:,5], color=baseValue.color_file[nFiles], marker='o', ls='None')
	plt.plot(bx, aFromFit, color=baseValue.color_file[nFiles], ls='--', label = ('fit : delta={:.2E}, xi={:.2E}, aIni={:.2E}'
																				 .format(result.params['delta'].value,result.params['xi'].value,result.params['aIni'].value)))
	plt.legend()

	return







#################################################
######### fit direct y(x) Olla euation  #########
#################################################
def fityollaglobal(nameDirectory, nFiles, nAllFile, baseValue) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)

	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt')
	V = np.loadtxt('V0.txt')

###############################################################################
	### Fit of the position y
		# function
	def ydataset(params, position) :
		U = params['U'].value
		delta = params['delta'].value
		yIn = params['yIni'].value
		r = params['radiusRBC'].value
		w = params['width'].value
		y0 = params['y0'].value
		h = params['height'].value

		x = position
		return fuctionforfit.fityollasimple(x, U, delta, y0, yIn, w, r, h)

		# residual function
	def objective(params, position, data, error) :
		residual = (data - ydataset(params, position))/error
		return residual.flatten()

		# constant problem
			# cte
	radius = 2.84e-6
	width = np.mean(datGlobal[:,1])
	yIni = datGlobal[0,2] ; yIniLim = datGlobal[0,3]
	if V[1]<1e-7 :
		V[1]=0

			# plot
	fig1 = plt.figure('fit y(x) by Olla ({})'.format(baseValue.name_file))
	ax1 = fig1.add_subplot(111)
	plt.xlabel('x (m)') ; plt.ylabel('y (m)')
	ax1.errorbar(datGlobal[:, 0], datGlobal[:, 2], yerr=datGlobal[:, 3], fmt='bo')

	fig2 = plt.figure('U in function of W')
	ax2 = fig2.add_subplot(111)
	plt.title('o for delta=1 and x for delta=2')
	plt.xlabel('w (m)') ; plt.ylabel('U')

###############################################################################
	### delta = 1, fit parameter U
		# Mise en place parameter
	fitParam0 = lm.Parameters()
	fitParam0.add( 'delta', 1, vary=False)
	fitParam0.add( 'U', 0, min=0, max=100)
	fitParam0.add( 'yIni' , yIni, min=yIni-yIniLim, max=yIni+yIniLim )
	fitParam0.add( 'width', width, vary=False)
	fitParam0.add( 'radiusRBC', radius, vary=False )
	fitParam0.add( 'height', baseValue.distance[0,9], vary=False)
	fitParam0.add( 'y0', V[1], vary=False)

		# Calcul fit
	result0 = lm.minimize(objective, fitParam0, method='differential_evolution',
						  args=(datGlobal[:,0],datGlobal[:,2], datGlobal[:,3]), nan_policy='omit')

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result0.params)

	### Plot
	yFromFit = ydataset(result0.params, datGlobal[:,0])
	ax1.plot(datGlobal[:,0], yFromFit, 'r-', label = ('fit : U={:.2E}, yIni={:.2E}'
													  .format(result0.params['U'].value, result0.params['yIni'].value)))
	plt.legend()
	ax2.plot(width, result0.params['U'].value, 'o', markersize=20)

###############################################################################
	### delta = 2, fit parameter U
		# Mise en place parameter
	fitParam1 = lm.Parameters()
	fitParam1.add('delta', 2, vary=False)
	fitParam1.add('U', 0, min=0, max=100)
	fitParam1.add('yIni', yIni, min=yIni - yIniLim, max=yIni + yIniLim)
	fitParam1.add('width', width, vary=False)
	fitParam1.add('radiusRBC', radius, vary=False)
	fitParam1.add( 'height', baseValue.distance[0,9], vary=False)
	fitParam1.add( 'y0', V[1], vary=False)

		# Calcul fit
	result1 = lm.minimize(objective, fitParam1, method='differential_evolution',
						 args=(datGlobal[:, 0], datGlobal[:, 2], datGlobal[:, 3]), nan_policy='omit')

	### Report fit
	lm.report_fit(result1.params)

	### Plot
	yFromFit = ydataset(result1.params, datGlobal[:, 0])
	ax1.plot(datGlobal[:, 0], yFromFit, 'g-', label=('fit : U={:.2E}, yIni={:.2E}'
													 .format(result1.params['U'].value, result1.params['yIni'].value)))
	plt.legend()
	ax2.plot(width, result1.params['U'].value, 'x', markersize=20)



	return





####################################
######### fit y=f(tN)  ##########
####################################
def fitytNqin(nameDirectory, nFiles, nAllFile, baseValue) :
		### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)

	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt')

	### Fit of the position a
	def datafit(params, position) :
		alpha = params['alpha'].value
		beta = params['beta'].value
		yIni = params['yIni'].value
		width = params['width'].value

		tN = position
		return fuctionforfit.fitytN(tN, alpha, beta, yIni, width)

		### residual function
	def objective(params, position, data, error) :
		residual = (data - datafit(params, position))/error
		return residual.flatten()

	### Calcul delta, xi et yIni optimal with the method differential_evolution'
		# Mise en place parameter
	r = 2.8e-6
	a = datGlobal[:,2] / r
	errA = datGlobal[:,3] / r
	width = np.mean(datGlobal[:, 1]) / r
	tN = datGlobal[:,17]

	fitParam = lm.Parameters()
	fitParam.add( 'alpha', 1, min=0, max=10)
	fitParam.add( 'beta', 0.1, min=1e-20, max=0.1)
	fitParam.add( 'yIni', a[0], min=a[0]-errA[0], max=a[0]+errA[0] )
	fitParam.add( 'width', width, vary=False)

		# Calcul fit
	result = lm.minimize( objective, fitParam, method='leastsq', args=(tN, a, errA), nan_policy='omit' )

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot fit
	plt.figure('fit alpha and beta for y(tN) ({})'.format(baseValue.name_file))
	plt.xlabel('x/w') ; plt.ylabel('y/w')
	testTN = np.linspace(0, 40000, 1000000)
	yFromFit = datafit(result.params, testTN)
	plt.errorbar(tN, a, yerr=errA, color=baseValue.color_file[nFiles], marker='o', ls='None')
	plt.plot(testTN, yFromFit, color=baseValue.color_file[nFiles], ls='--', label = ('fit : alpha={:.2E}, beta={:.2E}, yIni={:.2E}'
																				 .format(result.params['alpha'].value,result.params['beta'].value,result.params['yIni'].value)))
	plt.plot(np.array([0,100000]), np.array([width, width]))
	plt.legend()

	return







####################################
######### fit log(y(tN))  ##########
####################################
def fitytNglobal(nameDirectory, nFiles, nAllFile, baseValue) :
		### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)

	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt')

	### Fit of the position a
	def datafit(params, position) :
		alpha = params['alpha'].value
		y0 = params['y0'].value
		beta = params['beta'].value
		tN = position
		return fuctionforfit.fitytN(tN, alpha, beta, y0)

		### residual function
	def objective(params, position, data, error) :
		residual = (data - datafit(params, position))/error
		return residual.flatten()

	### Calcul delta, xi et yIni optimal with the method differential_evolution'
		# Mise en place parameter
	width = np.mean(datGlobal[:, 1])
	tN = datGlobal[:,17]

	fitParam = lm.Parameters()
	fitParam.add( 'alpha', 2, min=0, max=10)
	fitParam.add( 'beta', 0.1, min=1e-20, max=0.1)
	fitParam.add( 'y0', 0, min=0, max=width )

		# Calcul fit
	result = lm.minimize( objective, fitParam, method='differential_evolution', args=(tN, datGlobal[:,2], datGlobal[:,3]), nan_policy='omit' )

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot fit
	plt.figure('fit alpha and beta log(y)=log(f(t)) for {}'.format(baseValue.name_file))
	plt.xlabel('x/w') ; plt.ylabel('y/w')
	yFromFit = datafit(result.params, tN)
	plt.errorbar(tN, datGlobal[:,2], yerr=datGlobal[:,3], color=baseValue.color_file[nFiles], marker='o', ls='None')
	plt.plot(tN, yFromFit, color=baseValue.color_file[nFiles], ls='--', label = ('fit : alpha={:.2E}, beta={:.2E}, y0={:.2E}'
																				 .format(result.params['alpha'].value,result.params['beta'].value,result.params['y0'].value)))
	plt.legend()

	return







####################################
######### fit log(y(tN))  ##########
####################################
def fitlogytN(nameDirectory, nFiles, nAllFile, baseValue) :
		### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)

	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt')

	### Fit of the position a
	def datafit(params, position) :
		alpha = params['alpha'].value
		logBeta = params['logbeta'].value
		logt = position
		return fuctionforfit.fitlogytN(logt, alpha, logBeta)

		### residual function
	def objective(params, position, data, error) :
		residual = (data - datafit(params, position))/error
		return residual.flatten()

	### Calcul delta, xi et yIni optimal with the method differential_evolution'
		# Mise en place parameter
	width = np.mean(datGlobal[:, 1])
	r = 2.84e-6

	logT = np.log10(datGlobal[:,17])
	logY = np.log10( (datGlobal[:,2] - datGlobal[0,2]) / r )
	logYnW = np.log10(datGlobal[:,2] / width)
	logYnR = np.log10(datGlobal[:,2] / r)
	logYerror = np.log10(datGlobal[:,3])

	fitParam = lm.Parameters()
	fitParam.add( 'alpha', 2, min=0, max=5)
	fitParam.add( 'logbeta', -5, min=-15, max=10)

		# Calcul fit
	result = lm.minimize( objective, fitParam, method='differential_evolution', args=(logT, logY, logYerror), nan_policy='omit' )

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot fit
	plt.figure('fit alpha and logbeta for log(y(log(x))) ({})'.format(baseValue.name_file))
	plt.xlabel('log(tN)') ; plt.ylabel('log(y-yIni)')
	yFromFit = datafit(result.params, logT)
	plt.errorbar(logT, logY, yerr=logYerror, color=baseValue.color_file[nFiles], marker='o', ls='None')
	plt.plot(logT, logYnR, color=baseValue.color_file[nFiles], marker='x', ls='None')
	plt.plot(logT, logYnW, color=baseValue.color_file[nFiles], marker='+', ls='None')
	plt.plot(logT, yFromFit, color=baseValue.color_file[nFiles], ls='--', label = ('fit : alpha={:.2E}, logbeta={:.2E}'
																				 .format(result.params['alpha'].value,result.params['logbeta'].value)))
	plt.legend()

	return










####################################
######### fit log(y(tN))  ##########
####################################
def fittestglobal(nameDirectory, nFiles, nAllFile, baseValue) :
		### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)

	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt')


	### Plot
		# Value
	w = np.mean(datGlobal[:, 1])
	r = 2.84e-6

	t = datGlobal[:,17]
	y = datGlobal[:,2] - datGlobal[0,2]

		# Plot fit
	fig = plt.figure('plot test {}'.format(baseValue.name_file))

	fig.add_subplot(221)
	plt.xlabel('t') ; plt.ylabel('y-yIni')
	plt.plot(t, y, color=baseValue.color_file[nFiles], marker='o', ls='None', markersize=10)

	fig.add_subplot(222)
	plt.xlabel('t') ; plt.ylabel('(y-yIni)\\w')
	plt.plot(t, y/w, color=baseValue.color_file[nFiles], marker='+', ls='None', markersize=10)

	fig.add_subplot(223)
	plt.xlabel('t') ; plt.ylabel('(y-yIni)\\r')
	plt.plot(t, y/r, color=baseValue.color_file[nFiles], marker='+', ls='None', markersize=10)

	fig.add_subplot(224)
	plt.xlabel('beta : (y-y0)\\t^0.5');plt.ylabel('y-yIni')
	plt.plot(t, y / t**0.5, color=baseValue.color_file[nFiles], marker='+', ls='None', markersize=10)

	return










#############################
######### fit y(t)  #########
#############################
def fityx(nameDirectory, nFiles, nAllFile, baseValue) :
	### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)

	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt')

	### Fit of the position a
	def datafit(params, position):
		delta = params['delta'].value
		beta = params['beta'].value
		yIni = params['yIni'].value
		width = params['width'].value
		radius = params['radius'].value

		t = position
		return fuctionforfit.fityx(t, delta, beta, yIni, width, radius)

	### residual function
	def objective(params, position, data, error):
		residual = (data - datafit(params, position)) / error
		return residual.flatten()

	### Calcul delta, xi et yIni optimal with the method differential_evolution'
	# Mise en place parameter
	width = np.mean(datGlobal[:, 1])
	r = 2.84e-6

	x = datGlobal[:, 0]
	y = datGlobal[:, 2]
	error = datGlobal[:, 3]

	fitParam = lm.Parameters()
	fitParam.add('delta', 2, min=0, max=10)
	fitParam.add('beta', 1, min=1e-10, max=10)
	fitParam.add('yIni', y[0], min=y[0]-error[0], max=y[0]+error[0])
	fitParam.add('width', width, vary=False)
	fitParam.add('radius', r, vary=False)

	# Calcul fit
	result = lm.minimize(objective, fitParam, method='differential_evolution', args=(x[1:], y[1:], error[1:]),
						 nan_policy='omit')

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot fit
	plt.figure('fit delta and beta for y(x) ({})'.format(baseValue.name_file))
	plt.xlabel('x') ; plt.ylabel('y')
	yFromFit = datafit(result.params, x[1:])
	plt.errorbar(x, y, yerr=error, color=baseValue.color_file[nFiles], marker='o', ls='None')
	plt.plot(x[1:], yFromFit, color=baseValue.color_file[nFiles], ls='--',
			 label=('fit : delta={:.2E}, beta={:.2E}'.format(result.params['delta'].value, result.params['beta'].value)))
	plt.legend()

	return









##########################################
######### fit direct lambda xi  ##########
##########################################
def fitydirect(nameDirectory, nFiles, nAllFile, baseValue) :
		### Change of directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)

	### Lecture data
	datGlobal = np.loadtxt('globalAnalysisData.txt')

	### Fit of the position a
	def datafit(params, position) :
		lamb = params['lamb'].value
		yIni = params['yIni'].value
		xi = params['xi'].value
		tN = position
		return fuctionforfit.fitdirect(tN, lamb, xi, yIni)

		### residual function
	def objective(params, position, data, error) :
		residual = (data - datafit(params, position))/error
		return residual.flatten()

	### Calcul delta, xi et yIni optimal with the method differential_evolution'
		# Mise en place parameter
	r = 2.84e-6
	tN = datGlobal[:,17]
	y = datGlobal[:,2] / r
	yError = datGlobal[:,3] / r

	fitParam = lm.Parameters()
	fitParam.add( 'lamb', 3, min=0, max=10)
	fitParam.add( 'xi', 0.1, min=1e-20, max=1)
	fitParam.add( 'yIni', y[0], min=y[0]-yError[0], max=y[0]+yError[0] )

		# Calcul fit
	result = lm.minimize( objective, fitParam, method='differential_evolution', args=(tN, y, yError), nan_policy='omit' )

	### Report fit
	print('-----------   {}   -------------'.format(nAllFile[nFiles]))
	print('	the result for yfit are : ')
	lm.report_fit(result.params)

	### Plot fit
		# Plot fit
	plt.figure('fit lamb and xi : y = f(tN) for {}'.format(baseValue.name_file))
	plt.xlabel('normalized time') ; plt.ylabel('normalized y (y/R0)')
	yFromFit = datafit(result.params, tN)
	plt.errorbar(tN, y, yerr=yError, color=baseValue.color_file[nFiles], marker='o', ls='None', markersize=10)
	plt.plot(tN, yFromFit, color=baseValue.color_file[nFiles], ls='--', linewidth=4, label = ('fit : alpha={:.2E}, xi={:.2E}, yIni={:.2E}'
																				 .format(result.params['lamb'].value,result.params['xi'].value,result.params['yIni'].value)))
	plt.legend(fontsize=20)

		# Plot xi
	fig = plt.figure('xi in function of lambda, tN and y*')
	lamb = result.params['lamb'].value

			# xi in function tN
	ax1 = fig.add_subplot(121)
	plt.xlabel('normalized time') ; plt.ylabel('xi')
	ax1.plot(tN, (y**lamb-y[0]**lamb) / (lamb * tN), linewidth=4, color=baseValue.color_file[nFiles])

			# xi in function y
	ax2 = fig.add_subplot(122)
	plt.xlabel('y') ; plt.ylabel('xi')
	ax2.plot(y, (y ** lamb - y[0] ** lamb) / (lamb * tN), linewidth=4, color=baseValue.color_file[nFiles])


	return


