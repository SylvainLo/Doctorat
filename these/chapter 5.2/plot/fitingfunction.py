import numpy as np

import scipy.integrate as sci
import lmfit as lm

import matplotlib.pyplot as plt

import basefunction as bf




def zfit(xexp, P, vexp) :
	### calcul position z
		# function
	def dzdx(Z, X, Delta, Xi, H, R) :
		return Xi*R**(Delta+1) * ( H - Z) / ( (H*Z - Z**2/2 - R**2/6) * Z**Delta )

		# resolution position z (odeint)
	zth = sci.odeint( dzdx, P['zIni'].value, xexp,
					  args=(P['delta'].value, P['xi'].value, P['h'].value, P['rRBC'].value) )
	zShape = np.size( zth ) ; zth = np.reshape( zth, zShape )

	### L
	pt = vexp * (P['h'].value*zth - zth**2/2 - P['rRBC'].value**2/6)
	Lth = np.ones(np.shape(zth))
	Lth[1:] = sci.cumtrapz(pt, xexp)
	Lth[0] = 0

	return Lth


### residual function
def objective( params, position, data, velocity ) :
	resid = np.abs( data - zfit(position, params, velocity) )
	return resid.flatten()


### Calcul delta, xi et yIni optimal with the method differential_evolution
def fitingresult(l, p, zIni, vel):
	# basic value
	hExp = p[0,2]*1e-6 / 2
	r0 = 2.8e-6

	# Mise en place parameter
	fitParam = lm.Parameters( )
	fitParam.add( 'delta', 1.3, vary=False )
	fitParam.add( 'xi', 1e-8, min=1e-10, max=1 )
	fitParam.add( 'zIni', zIni, min=r0, max=hExp )
	fitParam.add( 'h', hExp, vary=False )
	fitParam.add( 'rRBC', r0, vary=False )

	# Calcul fit
	result = lm.minimize( objective, fitParam, method='differential_evolution', args=(p[:,1]*1e-6, l, vel), nan_policy='omit' )

	# Report fit
	print( '	the result for yfit are : ' )
	lm.report_fit( result.params )

	return result.params




def zfit2(xexp, P, vexp) :
	### calcul position z
		# function
	def dzdx(Z, X, Delta, Xi, H, R) :
		return Xi*R**(Delta+1) * ( H - Z) / ( (H*Z - Z**2/2 - R**2/6) * Z**Delta )

		# resolution position z (odeint)
	zth = sci.odeint( dzdx, P['zIni'].value, xexp,
					  args=(P['delta'].value, P['xi'].value, P['h'].value, P['rRBC'].value) )
	zShape = np.size( zth ) ; zth = np.reshape( zth, zShape )

	### L
	pt = 6/(3*P['h'].value**2-P['rRBC'].value**2) * vexp * (P['h'].value*zth - zth**2/2 - P['rRBC'].value**2/6)

	V = np.ones(np.shape( zth ))
	V[1:] = (1/xexp[1:]) * sci.cumtrapz(pt, xexp)
	V[0] = pt[0]

	return V


### Calcul delta, xi et yIni optimal with the method differential_evolution
def fastfitingresult(l, p, zIni, vel):
	# basic value
	hExp = p[0,2]*1e-6 / 2
	r0 = 2.8e-6

	# Mise en place parameter
	fitParam = lm.Parameters( )
	fitParam.add( 'delta', 1.3, vary=False )
	fitParam.add( 'xi', 1e-8, min=1e-10, max=100 )
	fitParam.add( 'zIni', zIni, min=r0, max=hExp )
	fitParam.add( 'h', hExp, vary=False )
	fitParam.add( 'rRBC', r0, vary=False )

	# Calcul fit
	result = lm.minimize( objective, fitParam, method='leastsq', args=(p[:,1]*1e-6, l, vel), nan_policy='omit' )

	# Report fit
	print( '	the result for yfit are : ' )
	lm.report_fit( result.params )

	return result.params

