import numpy as np

import matplotlib.ticker as plttick

import lmfit as lm

import functionfit




###########################################
######### function to fit a ratio #########
###########################################
### Variable
R0 = 2.8e-6
N = 7  # odd number








###########################################
######### function to fit a ratio #########
###########################################
def fitaxtotal(data, fig, p, k) :
	### Plot
	ax = fig.add_subplot(3,2,k)
	ax.set_xlabel(r'$\tilde{x}$')
	ax.set_ylabel(r'$\tilde{y}$')


	### data
	width = np.mean(data['data'][:, 2])
	a = data['data'][:, 6]
	aErr = (data['data'][:, 7] + data['data'][:, 8]) / 2
	b = data['data'][:, 0] / width
	bErr = data['data'][:, 1] / width

	Rstar = R0 / width ; rExpStar = Rstar
	hStar = data['baseData'].distance[0, 9] / width


	### using optimal solution
		# creating list parameter
	fitParam = lm.Parameters()
	fitParam.add('delta', p[0], vary=False)
	fitParam.add('xi', p[1], vary=False)
	fitParam.add('n', N, vary=False)
	fitParam.add('radiusRBC', Rstar, vary=False)
	fitParam.add('rExpRBC', rExpStar, vary=False)
	fitParam.add('yIni', a[0], min=a[0]-aErr[0], max=a[0] )
	fitParam.add('height', hStar, vary=False )

		# calculate
	result = lm.minimize(objective, fitParam, method='leastsq', nan_policy='omit',
						  args=(b, a, aErr))
	lm.report_fit(result.params)

		# curve
	bFit = np.linspace(0, b.max(), 101)
	fit = ydataset(result.params, bFit)


	### plot
	ax.errorbar(b, a, xerr=bErr, yerr=[data['data'][:, 7], data['data'][:, 8]], ls='None', color='r', elinewidth=1,
				label='experimental data')
	ax.plot(bFit, fit, 'k', linewidth=1)

	ax.yaxis.set_major_locator(plttick.MultipleLocator(base=0.1))

	ax.text(0.95, 0.05, r'$w$ = {:.1f} $\mu $m'.format(width * 1e6), horizontalalignment='right',
			verticalalignment='bottom', transform=ax.transAxes)




	return fig






######################################
######### function residual  #########
######################################
### for fit delta xi ###
	### Fit of the position y
def ydataset(params, position) :
	delta = params['delta'].value
	n = params['n'].value
	xi = params['xi'].value
	rExp = params['rExpRBC'].value
	yIni = params['yIni'].value
	h = params['height'].value
	rThe = params['radiusRBC'].value

	x = position
	return functionfit.afit(x, delta, xi, rExp, yIni, h, rThe, n)

	### residual function
def objective(params, position, a, aErr) :
	residual = np.array( a  - ydataset(params, position) ) / aErr
	return residual.flatten()









#######################################
######### fit light and heavy #########
#######################################
def fitaxtotalhl(data1, data2, data3, fig, p) :

### Plot ###
	# ax for heavy light
	ax = fig.add_axes([0.15,0.15,0.80,0.80])
	ax.set_xlabel(r'$\tilde{x}$') ; ax.set_ylabel(r'$\tilde{y}$')
	ax.set_ylim(0.39, 0.92)
	colorPlot = ['r', 'b']
	namePlot = ['higher density', 'less density']
	c=0


	# ax for rigidified
	axint = fig.add_axes([0.56, 0.16, 0.38, 0.35])
	axint.set_xlabel(r'$\tilde{x}$')
	axint.xaxis.set_label_position('top') ; axint.xaxis.set_ticks_position('top')
	axint.set_ylabel(r'$\tilde{y}$')


### data ###
	for data in [data1, data2] :
	### data
		width = np.mean(data['data'][:, 2])
		a = data['data'][:, 6]
		aErr = (data['data'][:, 7] + data['data'][:, 8]) / 2
		b = data['data'][:, 0] / width
		bErr = data['data'][:, 1] / width

		Rstar = R0 / width ; rExpStar = Rstar
		hStar = data['baseData'].distance[0, 9] / width


	### using optimal solution
		# creating list parameter
		fitParam = lm.Parameters()
		fitParam.add('delta', p[0], vary=False)
		fitParam.add('xi', p[1], min=0, max=1)
		fitParam.add('n', N, vary=False)
		fitParam.add('radiusRBC', Rstar, vary=False)
		fitParam.add('rExpRBC', rExpStar, vary=False)
		fitParam.add('yIni', a[0], min=1e-6/width, max=1 )
		fitParam.add('height', hStar, vary=False )

		# calculate
		result = lm.minimize(objective, fitParam, method='differential_evolution', nan_policy='omit',
						  args=(b, a, aErr))
		lm.report_fit(result.params)

		# curve
		bFit = np.linspace(0, b.max(), 101)
		fit = ydataset(result.params, bFit)


	### plot
		ax.plot(b, a, ls='None', marker='o', markersize=3, color=colorPlot[c], label=namePlot[c])
		ax.plot(bFit, fit, color=colorPlot[c], ls='--' ,linewidth=1)
		c = c+1


### fit from all data ###
	### variable
	width = (np.mean(data1['data'][:, 2]) + np.mean(data2['data'][:, 2]))/2
	a0 = (data1['data'][0, 6] + data2['data'][0, 6])/2

	Rstar = R0 / width ; rExpStar = Rstar
	hStar = data['baseData'].distance[0, 9] / width

	### parameter
	fitParam = lm.Parameters()
	fitParam.add('delta', p[0], vary=False)
	fitParam.add('xi', p[1], vary=False)
	fitParam.add('n', N, vary=False)
	fitParam.add('radiusRBC', Rstar, vary=False)
	fitParam.add('rExpRBC', rExpStar, vary=False)
	fitParam.add('yIni', a0, vary=False )
	fitParam.add('height', hStar, vary=False )

	### curve
	bFit = np.linspace(0, b.max(), 101)
	fit = ydataset(fitParam, bFit)


### fit rigified ###
	fitaxtotalrigidified(data3, axint, p)


	### plot
	ax.plot(bFit, fit, 'k', ls='-' ,linewidth=1)

	ax.text(0.05, 0.85, r'$w$ = {:.1f} $\mu $m'.format(width * 1e6), horizontalalignment='left',
			verticalalignment='bottom', transform=ax.transAxes)

	return fig







#######################################
######### fit light and heavy #########
#######################################
def fitaxtotalrigidified(data, ax, p) :
### data ###
	### data
	width = np.mean(data['data'][:, 2])
	a = data['data'][:, 6]
	aErr = (data['data'][:, 7] + data['data'][:, 8]) / 2
	b = data['data'][:, 0] / width
	bErr = data['data'][:, 1] / width

	Rstar = R0 / width ; rExpStar = Rstar
	hStar = data['baseData'].distance[0, 9] / width

	### parameter
	fitParam = lm.Parameters()
	fitParam.add('delta', p[0], vary=False)
	fitParam.add('xi', p[1], vary=False)
	fitParam.add('n', N, vary=False)
	fitParam.add('radiusRBC', Rstar, vary=False)
	fitParam.add('rExpRBC', rExpStar, vary=False)
	fitParam.add('yIni', a[0], vary=False)
	fitParam.add('height', hStar, vary=False)

	### curve
	bFit = np.linspace(0, b.max(), 101)
	fit = ydataset(fitParam, bFit)


	### plot
	ax.errorbar(b, a, xerr=bErr, yerr=[data['data'][:, 7], data['data'][:, 8]], ls='None', elinewidth=1,
			color='r', label='rigidified RBC')
	ax.plot(bFit, fit, 'k', ls='-' ,linewidth=1)


	return



