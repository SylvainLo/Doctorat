import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plttick
import corner

import lmfit as lm
import emcee

import os

import functionfit

R0 = 2.8e-6
N = 5


def plotCorrelation(dat) :
	### plot
	figC = plt.figure( 'correlation', figsize=[5.3, 7] )
	figC.subplots_adjust( 0.09, 0.08, 0.96, 1, 0.45, 0.45 )

	### creation axis
	axC1 = figC.add_subplot( 421 )
	axC1.set_ylabel( r'$\delta$' ) ; axC1.set_xlabel( r'$ \dot{\gamma_{e}} $ (s-1)' )

	axC2 = figC.add_subplot( 422 )
	axC2.set_ylabel( r'$\xi$' ) ; axC2.set_xlabel( r'$ \dot{\gamma_{e}} $ (s-1)' )

	axC3 = figC.add_subplot( 423 )
	axC3.set_ylabel( r'$\delta$' ) ; axC3.set_xlabel( r'$ v_{max} $ (mm.s-1)' )

	axC4 = figC.add_subplot( 424 )
	axC4.set_ylabel( r'$\xi$' ) ; axC4.set_xlabel( r'$ v_{max} (mm.s-1)$' )

	axC5 = figC.add_subplot( 425 )
	axC5.set_ylabel( r'$\delta$' ) ; axC5.set_xlabel( r'$ w/R_0 $' )

	axC6 = figC.add_subplot( 426 )
	axC6.set_ylabel( r'$\xi$' ) ; axC6.set_xlabel( r'$ w/R_0 $' )

	axC7 = figC.add_subplot( 427 )
	axC7.set_ylabel( r'$\delta$' ) ; axC7.set_xlabel( r'$ w/h $' )

	axC8 = figC.add_subplot( 428 )
	axC8.set_ylabel( r'$\xi$' ) ; axC8.set_xlabel( r'$ w/h $' )

	### movie used
	nMov = [0,1,2,3,7,8]


	### utilisation data
	for i in nMov :
		# extract data
		datAna = dat['{}'.format(i)]
		w = np.mean( datAna['data'][:, 2])
		h = datAna['baseData'].distance[0, 9]
		v = datAna['vmax']
		shear = v/w
		aspect = w/h
		conf = w/R0

		# measure delta and xi
		val = fitcurve(datAna, i)
		delta = val['delta'].value
		xi = val['xi'].value

		# plot
		axC1.plot(shear, delta, 'ro')
		axC2.plot(shear, xi, 'ro')
		axC3.plot(v*1000, delta, 'ro')
		axC4.plot(v*1000, xi, 'ro')
		axC5.plot(conf, delta, 'ro')
		axC6.plot(conf, xi, 'ro')
		axC7.plot(aspect, delta, 'ro')
		axC8.plot(aspect, xi, 'ro')



	return figC







def fitcurve(data, ii) :
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
	fitParam.add('delta', 1, min=0, max=10)
	fitParam.add('xi', 0.01, min=1e-10, max=1)
	fitParam.add('n', N, vary=False)
	fitParam.add('radiusRBC', Rstar, vary=False)
	fitParam.add('rExpRBC', rExpStar, vary=False)
	fitParam.add('yIni', a[0], min=1e-6/width, max=1 )
	fitParam.add('height', hStar, vary=False )

		# calculate first estimation results
	result = lm.minimize(objective, fitParam, method='leastsq', nan_policy='omit',
						  args=(b, a, aErr))
	lm.report_fit(result.params)
	print(result.params)



	# 	# use of Monte Carlo approximation to have error on fit
	# 		# add a new parameter to create noise in the data
	# result.params.add( 'noise', value=0.1, min=0.01, max=1.5 )
	# resultMini = lm.Minimizer( lnprob, result.params, fcn_args=(b, a, aErr))
	# res = resultMini.emcee( burn=100, steps=500, thin=20, params=result.params )
	#
	# 		# plot and show result
	# lm.report_fit( res.params )
	# plt.figure('monte_carlo_{}'.format(ii))
	# corner.corner( res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
	#
	# 		# extract interesting data (best value and quartile)
	# highest_prob = np.argmax( res.lnprob )
	# hp_loc = np.unravel_index( highest_prob, res.lnprob.shape )
	# mle_soln = res.chain[hp_loc]
	#
	#
	# quDelta = np.percentile( res.flatchain['delta'], [2.28, 15.9, 50, 84.2, 97.7] )
	# print(quDelta)
	# print("1 sigma spread", 0.5*(quDelta[3]-quDelta[1]))
	#
	# quXi= np.percentile( res.flatchain['xi'], [2.28, 15.9, 50, 84.2, 97.7] )
	# print(quXi)
	# print("1 sigma spread", 0.5*(quXi[3]-quXi[1]))


	return(result.params)





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

	### residual function emcee
def lnprob(params, position, data, error) :
	noise = params['noise'].value
	residual = (data - ydataset(params, position)) / error
	logProb = -0.5 * np.sum( (residual/noise)**2 + np.log(2* np.pi* noise**2) )
	return logProb