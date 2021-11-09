import numpy as np

import matplotlib.pyplot as plt
from math import cosh, pi

import lmfit as lm

import os

import fuctionforfit




#########################################
######### function def all data #########
#########################################
def dicalldata(nameDirectory, nAllFile, sizeAllFile) :
	### Creation variable with all global data
	dicAllData = {}
	V0 = {}

	for i in range(sizeAllFile) :
		# Change of directory
		subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[i]) 
		os.chdir(subDirectoryVideo)

		# Lecture data
		datGlobal = np.loadtxt('globalAnalysisData.txt') 
		vMax = np.loadtxt('V0.txt')

		# remplissage dictionnaire
		dicAllData['data{}'.format(i)] = datGlobal
		V0['V0-{}'.format(i)] = vMax

	os.chdir(nameDirectory)

	return dicAllData, V0





###########################################
######### function to fit a ratio #########
###########################################
def fitaxtotal(nameDirectory, nAllFile, sizeAllFile, dicValue) :
	### Creation variable with all global data
	(dicAllData, V0) = dicalldata(nameDirectory, nAllFile, sizeAllFile)

	### Variable
	R0 = 2.84e-6



###############################################################################
### delta = constant , xi constant ###
	### Fit of the position y with different xi
	def ydataset1(params, position, n) :
		delta = params['delta'].value
		xi = params['xi'].value
		y0 = params['y0_{}'.format(n)].value
		r = params['radiusRBC_{}'.format(n)].value
		yIni = params['yIni_{}'.format(n)].value
		w = params['width_{}'.format(n)].value
		h = params['height_{}'.format(n)].value

		x = position
		return fuctionforfit.afit(x, delta, xi, y0, yIni, w, r, h)


	### residual function
	def objective1(params) :
		residual = np.array([], dtype=np.float)
		for k in range(sizeAllFile) :
			a = dicAllData["data{}".format(k)][:,4]
			aErr = dicAllData["data{}".format(k)][:,5]
			position = dicAllData["data{}".format(k)][:,0] / params['width_{}'.format(k)].value

			parResidual = np.array( a  - ydataset1(params, position, k) ) / aErr
			residual = np.concatenate((residual, parResidual))
		return residual


	### Calculate delta, xi et yIni optimal with the method differential_evolution
		# creating list parameter
	fitParam0 = lm.Parameters()
	fitParam0.add( 'delta', 1, min=0, max=5)
	fitParam0.add( 'xi', 1e-4, min=1e-10, max=10)

	for i in range(sizeAllFile) :
		width = np.mean(dicAllData["data{}".format(i)][:,1])
		y0Star = V0['V0-{}'.format(i)][1] / width
		yIniStar = (dicAllData["data{}".format(i)][0, 2])/ width
		yIniLimStar = dicAllData["data{}".format(i)][0, 3] / width
		hStar = dicValue["variableForFile{}".format(i)].distance[0,9] / width

		fitParam0.add( 'radiusRBC_{}'.format(i), R0/width, vary=False)
		fitParam0.add( 'y0_{}'.format(i), y0Star, vary=False)
		fitParam0.add( 'yIni_{}'.format(i), yIniStar, min=yIniStar-yIniLimStar, max=yIniStar+yIniStar )
		fitParam0.add( 'width_{}'.format(i), width, vary=False )
		fitParam0.add( 'height_{}'.format(i), hStar, vary=False )

		# Calculate fit
	result0 = lm.minimize(objective1, fitParam0, method='differential_evolution', nan_policy='omit')
	lm.report_fit(result0.params)

		# Plot data
	os.chdir(nameDirectory)
	fig = plt.figure('fit total delta et xi cte')
	ax = fig.add_subplot(111)
	plt.xlabel(r'$x$ (mm)') ; plt.ylabel(r'$y$ ($\mu$m)')
	plt.title(r'$\delta$ = {:.1f}, $\xi$ = {:.2}'.format(result0.params['delta'].value, result0.params['xi'].value))
	for i in range(sizeAllFile) :
		baseValue = dicValue["variableForFile{0}".format(i)]
		width = np.mean(dicAllData["data{0}".format(i)][:,1])
		xPlot = dicAllData["data{0}".format(i)][:,0]/width
		yFromFit = ydataset1(result0.params, xPlot, i)
		ax.plot(xPlot, dicAllData["data{0}".format(i)][:,4], color=baseValue.color_file[i], ls='None', marker='o',
				 markersize=1, alpha=0.6)
		ax.plot(xPlot, yFromFit, color=baseValue.color_file[i], ls=':', linewidth=1,
				 label = (r'$w$ = {:.1f} $\mu$m, $y_{{Ini}}$ = {:.1f}'.format(result0.params['width_{}'.format(i)].value*1e6,
																	   result0.params['yIni_{}'.format(i)].value)))
	plt.legend()
	plt.savefig('fit y xi et delta cte.pdf', frameon=False, bbox_inch='tight', orientation='landscape')




###############################################################################
### delta variable , xi variable ###
	### Fit of the position y
	def ydataset4( params, position ):
		delta = params['delta'].value
		xi = params['xi'].value
		y0 = params['y0'].value
		r = params['radiusRBC'].value
		yIni = params['yIni'].value
		w = params['width'].value
		h = params['height'].value

		x = position
		return fuctionforfit.afit(x, delta, xi, y0, yIni, w, r, h)


	### residual function
	def objective4( params, position, data, error ):
		residual = np.array((data-ydataset4(params, position)) / error)
		return residual.flatten()

	### Creating graph
	figW = plt.figure('delta <-> W et xi <-> W')
	ax1FigW = figW.add_subplot(1, 1, 1)
	ax2FigW = ax1FigW.twinx()

	figV0 = plt.figure('delta <-> V0 et xi <-> V0')
	ax1FigV0 = figV0.add_subplot(1, 1, 1)
	ax2FigV0 = ax1FigV0.twinx()

	figYIni = plt.figure('delta <-> yIni et xi <-> yIni')
	ax1FigYIni = figYIni.add_subplot(1, 1, 1)
	ax2FigYIni = ax1FigYIni.twinx()


	### Calculate delta, xi et yIni optimal with the method differential_evolution
	for i in range(sizeAllFile):
		# variable for 1 experiment
		baseValue = dicValue["variableForFile{}".format(i)]
		width = np.mean(dicAllData["data{}".format(i)][:, 1])
		y0 = V0['V0-{}'.format(i)][1] / width
		yInitial = dicAllData["data{}".format(i)][0,4]
		yIniLim = dicAllData["data{}".format(i)][0,5]
		hStar = dicValue["variableForFile{}".format(i)].distance[0,9] / width

		# Mise en place parameter
		fitParam4 = lm.Parameters()
		fitParam4.add( 'delta', 0, min=-5, max=5 )
		fitParam4.add( 'xi', 1e-5, min=1e-10, max=10 )
		fitParam4.add( 'radiusRBC', R0 / width, vary=False )
		fitParam4.add( 'y0', y0, vary=False )
		fitParam4.add( 'yIni', yInitial, min=yInitial-yIniLim, max=yInitial+yIniLim )
		fitParam4.add( 'width', width, vary=False )
		fitParam4.add( 'height', hStar, vary=False )

		# Calculate fit
		b = dicAllData["data{}".format(i)][:, 0] / width
		a = dicAllData["data{}".format(i)][:, 4]
		err = dicAllData["data{}".format(i)][:, 5]

		result4 = lm.minimize(objective4, fitParam4, method='differential_evolution', nan_policy='omit',
							  args=(b, a, err))
		print('----------------{}----------------'.format(nAllFile[i]))
		lm.report_fit(result4.params)

	### Plot
		# Plot all fit
		yFromFit = ydataset4(result4.params, b)
		plt.figure('fit global all data')
		plt.plot(b, a, color=baseValue.color_file[i], ls='None', marker='o', markersize=1, alpha=0.5)
		plt.plot(b, yFromFit, color=baseValue.color_file[i], ls=':', linewidth=1,
				 label=(r'fit : $\delta$={:.2}, $\xi$={:.2E}, $w$={:.1} $\mu$m'.format(result4.params['delta'].value,
																				result4.params['xi'].value,
																				result4.params['width'].value*1e6)))

		# Plot delta <-> xi
		plt.figure('delta <-> xi')
		plt.plot(result4.params['delta'].value, result4.params['xi'].value,
				 color=baseValue.color_file[i], ls='None', marker='o')

		# Plot delta <-> W et xi <-> W
		plt.figure('delta <-> W et xi <-> W')
		ax1FigW.plot(width, result4.params['delta'].value,
					 color=baseValue.color_file[i], ls='None', marker='o')
		ax2FigW.plot(width, result4.params['xi'].value,
					 color=baseValue.color_file[i], ls='None', marker='x')

		# Plot delta <-> V0 et xi <-> V0
		plt.figure('delta <-> V0 et xi <-> V0')
		ax1FigV0.plot(V0['V0-{}'.format(i)][0], result4.params['delta'].value,
					  color=baseValue.color_file[i], ls='None', marker='o')
		ax2FigV0.plot(V0['V0-{}'.format(i)][0], result4.params['xi'].value,
					  color=baseValue.color_file[i], ls='None', marker='x')

		# Plot delta <-> yIni et xi <-> yIni
		plt.figure('delta <-> yIni et xi <-> yIni')
		ax1FigYIni.plot(yInitial, result4.params['delta'],
						color=baseValue.color_file[i], ls='None', marker='o')
		ax2FigYIni.plot(yInitial, result4.params['xi'],
						color=baseValue.color_file[i], ls='None', marker='o')


	### Finalising plot
	os.chdir(nameDirectory)
	plt.figure('fit global all data')
	plt.xlabel(r'x (m)') ; plt.ylabel(r'y (m)')
	plt.legend(fontsize=6)
	plt.savefig('fit global all data.pdf', frameon=False, bbox_inch='tight', orientation='landscape')

	plt.figure('delta <-> xi')
	plt.xlabel(r'delta') ; plt.ylabel(r'xi')
	plt.legend(fontsize=6)
	plt.savefig('delta-xi.pdf', frameon=False, bbox_inch='tight', orientation='landscape')

	plt.figure('delta <-> W et xi <-> W')
	ax1FigW.set_xlabel(r'W en m'); ax1FigW.set_ylabel(r'delta')
	ax2FigW.set_ylabel(r'xi')
	plt.legend(fontsize=6)
	plt.title(r'$\delta$ = o et $\xi$ = x')
	plt.savefig('delta-W et xi-W.pdf', frameon=False, bbox_inch='tight', orientation='landscape')

	plt.figure('delta <-> V0 et xi <-> V0')
	ax1FigV0.set_xlabel(r'$V_{max}$ en m.s$^{-1}$'); ax1FigV0.set_ylabel(r'$\delta$')
	ax2FigV0.set_ylabel(r'$\xi$')
	plt.legend(fontsize=6)
	plt.title(r'$\delta$ = o et $\xi$ = x')
	plt.savefig('delta-V0 et xi-V0.pdf', frameon=False, bbox_inch='tight', orientation='landscape')

	plt.figure('delta <-> yIni et xi <-> yIni')
	ax1FigYIni.set_xlabel('YIni en m/s'); ax1FigYIni.set_ylabel('delta')
	ax2FigYIni.set_ylabel('xi')
	plt.legend(fontsize=6)
	plt.title('delta = o et xi = x')
	plt.savefig('delta-YIni et xi-YIni.pdf', frameon=False, bbox_inch='tight', orientation='landscape')
	return