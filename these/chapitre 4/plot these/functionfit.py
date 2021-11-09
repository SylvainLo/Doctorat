import numpy as np

import scipy.integrate as sci
import scipy.interpolate as sci2

from math import pi, cosh, sin, sinh

import lmfit as lm

import matplotlib.pyplot as plt

#######################################
######### fit light and heavy #########
#######################################
def fitymean(x, y, w, h, r, ax) :
### data ###
	### data
	aStar = x / w
	bStar = y / w
	rStar = r / w
	hStar = h / w
	N=3

	### parameter
	fitParam = lm.Parameters()
	fitParam.add('delta', 1.3, vary=False)
	fitParam.add('xi', 0.01, min=1e-5, max=10, vary=True)
	fitParam.add('yIni', bStar[0], min=0.1, max=0.9, vary=True)

	fitParam.add('n', N, vary=False)
	fitParam.add('radiusRBC', rStar, vary=False)
	fitParam.add('height', hStar, vary=False)

	# plt.figure('{}'.format(ax))
	# plt.plot(aStar, bStar)

	# calculate
	result = lm.minimize( objective, fitParam, method='differential_evolution', nan_policy='omit', args=(aStar, bStar) )
	lm.report_fit( result.params )

	### curve
	aFit = np.linspace(aStar.min(), aStar.max(), 101)
	fit = ydataset(result.params, aFit)

	# plt.figure( '{}'.format(ax) )
	# plt.plot( aFit, fit )


	### plot
	ax.plot((aFit*w)/1000, fit*w, 'r', ls='-' ,linewidth=1)


	return ax


######################################
######### function residual  #########
######################################
### for fit delta xi ###
	### Fit of the position y
def ydataset(params, position) :
	delta = params['delta'].value
	n = params['n'].value
	xi = params['xi'].value
	y0 = params['yIni'].value
	hf = params['height'].value
	rf = params['radiusRBC'].value

	xf = position
	return bfit(xf, delta, xi, y0, hf, rf, n)

	### residual function
def objective(params, position, b) :
	residual = np.array( np.abs(b  - ydataset(params, position)) )
	return residual.flatten()



###########################################
######### function to fit a ratio #########
###########################################

def bfit( aFit, deltaFit, xiFit, b0, hFit, rFit, nFit):
	### vector x
	aX = np.linspace(0, np.max(aFit)+1, 10000)

	### first method
		# ode
	def dadb( bD, aFit, deltaD, xiD, hD, rD, nD):
		""" we use the velociy profile from a rectangular channel with limn = n (included)
		hD half the height of the channel"""
		A = 0 ; B = 0

		for ii in range(1, nD+1, 2) :
			c1 = (-1)**(int((ii-1)/2)) / ii**3
			c2 = ii * pi / (2*hD)

			a1 = cosh(c2)

			A = A + ii*c1 * ( np.sinh(c2*(bD-1)) / a1 )
			B = B + c1/ii * ( rD - (np.cosh(c2*(bD-1))*sinh(c2*rD))/(c2*a1) ) * sin(c2*rD)

		cte = xiD * rD**2 * rD**(deltaD+1) * pi**2 / (bD**deltaD * hD**2)
		va = - cte * (A / B)
		return va

		# calcul position y
	bOde = sci.odeint(dadb, b0, aX, args=(deltaFit, xiFit, hFit, rFit, nFit))
	bOde = bOde[:, 0]

	### interpolation
	f = sci2.interp1d(aX, bOde)
	bFinal = f(aFit)

	return bFinal