import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plttick
import math

import lmfit as lm

import os

import functionfit





###########################################
######### function to fit a ratio #########
###########################################
def figPlot(data3, dir) :
	### change directory ###
	os.chdir( dir )

	### Fit ###
	### data
	# variable general
	R0 = 2.8e-6
	N = 7

	# particular variable
	width = np.mean( data3['data'][:, 2] )
	a = data3['data'][:, 6]
	aErr = (data3['data'][:, 7]+data3['data'][:, 8])/2
	b = data3['data'][:, 0]/width
	bErr = data3['data'][:, 1]/width

	rExpStar = R0/width
	hStar = data3['baseData'].distance[0, 9]/width
	Rstar = R0/width

	### fit
	# delta = 1, xi = variable
	fitParam1 = lm.Parameters( )
	fitParam1.add( 'delta', 1, vary=False )
	fitParam1.add( 'xi', 1e-4, min=1e-6, max=1 )
	fitParam1.add( 'n', N, vary=False )
	fitParam1.add( 'radiusRBC', Rstar, vary=False )
	fitParam1.add( 'rExpRBC', rExpStar, vary=False )
	fitParam1.add( 'yIni', a[0], min=0.5e-6/width, max=1 )
	fitParam1.add( 'height', hStar, vary=False )

	result1 = lm.minimize( objective1, fitParam1, method='differential_evolution', nan_policy='omit',
						   args=(b, a, aErr) )
	lm.report_fit( result1.params )

	# delta = 2 , xi variable
	fitParam2 = lm.Parameters( )
	fitParam2.add( 'delta', 2, vary=False )
	fitParam2.add( 'xi', 1e-4, min=1e-6, max=1 )
	fitParam2.add( 'n', N, vary=False )
	fitParam2.add( 'radiusRBC', Rstar, vary=False )
	fitParam2.add( 'rExpRBC', rExpStar, vary=False )
	fitParam2.add( 'yIni', a[0], min=0.5e-6/width, max=1 )
	fitParam2.add( 'height', hStar, vary=False )

	result2 = lm.minimize( objective1, fitParam2, method='differential_evolution', nan_policy='omit',
						   args=(b, a, aErr) )
	lm.report_fit( result2.params )

	# delta = 0 , xi variable
	fitParam0 = lm.Parameters( )
	fitParam0.add( 'delta', 0, vary=False )
	fitParam0.add( 'xi', 1e-4, min=1e-6, max=1 )
	fitParam0.add( 'n', N, vary=False )
	fitParam0.add( 'radiusRBC', Rstar, vary=False )
	fitParam0.add( 'rExpRBC', rExpStar, vary=False )
	fitParam0.add( 'yIni', a[0], min=0.5e-6/width, max=1 )
	fitParam0.add( 'height', hStar, vary=False )

	result0 = lm.minimize( objective1, fitParam0, method='differential_evolution', nan_policy='omit',
						   args=(b, a, aErr) )
	lm.report_fit( result0.params )

	# delta variable , xi variable
	fitParamVar = lm.Parameters( )
	fitParamVar.add( 'delta', 1, min=0, max=10 )
	fitParamVar.add( 'xi', 1e-4, min=1e-6, max=1 )
	fitParamVar.add( 'n', N, vary=False )
	fitParamVar.add( 'radiusRBC', Rstar, vary=False )
	fitParamVar.add( 'rExpRBC', rExpStar, vary=False )
	fitParamVar.add( 'yIni', a[0], min=1e-6/width, max=a.max( ) )
	fitParamVar.add( 'height', hStar, vary=False )

	resultVar = lm.minimize( objective1, fitParamVar, method='differential_evolution', nan_policy='omit',
							 args=(b, a, aErr) )
	lm.report_fit( resultVar.params )

	### plot ###
	### Initialisation plot
	# fig with data and fit
	fig = plt.figure( 'fit all data multiple graph', figsize=[4.5, 4.5] )
	ax = fig.add_axes( [0.15, 0.12, 0.80, 0.80] )
	ax.set_xlabel( r'$\tilde{x}$' ) ; ax.set_ylabel( r'$\tilde{y}$' )
	ax.set_ylim( 0.25, 0.95 )

	# ax with surface
	# axint = inset_axes(ax, width='50%', height='50%', loc=4)
	axint = fig.add_axes( [0.52, 0.14, 0.40, 0.40] )
	axint.set_xlabel( r'$\delta$' )
	axint.xaxis.set_label_position( 'top' ) ; axint.xaxis.set_ticks_position( 'top' )
	axint.set_ylabel( r'$log( \xi )$' )
	axint.set_yscale( 'log' )

	### Plot fit and curve
	# data and fit
	# value fit
	bFit = np.linspace( 0, b.max( ), 101 )
	fitDelta1 = ydataset1( result1.params, bFit )
	fitDelta2 = ydataset1( result2.params, bFit )
	fitDeltaVar = ydataset1( resultVar.params, bFit )
	fitDelta0 = ydataset1( result0.params, bFit )

	# plot
	l1 = ax.errorbar( b, a, xerr=bErr, yerr=[data3['data'][:, 7], data3['data'][:, 8]], ls='None', elinewidth=1,
					  color='r', zorder=5, label='experimental data' )
	print( l1 )
	l2, = ax.plot( bFit, fitDelta1, 'k--', linewidth=1, zorder=7, label='$\delta = 1$' )
	l3, = ax.plot( bFit, fitDelta2, 'k:', linewidth=1, zorder=8, label='$\delta = 2$' )
	l4, = ax.plot( bFit, fitDelta0, 'k-.', linewidth=1, zorder=9, label='$\delta = 0$' )
	l5, = ax.plot( bFit, fitDeltaVar, 'k-', linewidth=1, zorder=6,
				   label='$\delta = ${:.1f}'.format( resultVar.params['delta'].value ) )

	ax.legend( handles=[l1, l5, l4, l2, l3], loc=(0.02, 0.68), frameon=False, framealpha=None )

	# data residual
	# X and Y
	X = np.load( 'Xforresidual.npy' ) ; Y = np.load( 'Yforresidual.npy' )

	# plot
	residual = data3['residual']
	level = plttick.MaxNLocator( nbins=10 ).tick_values( residual.min( ), residual.max( ) )
	plot = axint.contourf( X, Y, residual, levels=level, cmap='viridis' )

	return fig






######################################
######### function residual  #########
######################################
### for fit delta xi ###
### Fit of the position y
def ydataset1( params, position ) :
	delta = params['delta'].value
	n = params['n'].value
	xi = params['xi'].value
	rExp = params['rExpRBC'].value
	yIni = params['yIni'].value
	h = params['height'].value
	rThe = params['radiusRBC'].value

	x = position
	return functionfit.afit( x, delta, xi, rExp, yIni, h, rThe, n )



### residual function
def objective1( params, position, a, aErr ) :
	residual = np.array( a-ydataset1( params, position ) )/aErr
	return residual.flatten( )

