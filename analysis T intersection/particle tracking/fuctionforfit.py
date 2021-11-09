import numpy as np

import scipy.integrate as sci
import scipy.interpolate as sci2

from math import pi, cosh, tanh, sinh

import matplotlib.pyplot as plt


#####################################
######### function to fit v #########
#####################################

def vrectangularfit(y, h, w, vmax, n) :
	A = 0 ; B = 0

	for i in range(0, n, 1) :
		AB1 = (-1)**i / (2*i+1)**3
		AB2 = cosh((2*i+1)*pi*w/h)
		A = A + AB1 * (1 - 1/AB2)

		B3 = np.cosh((2*i+1)*pi*(y-w)/h)
		B = B + AB1 * (1 - B3/AB2)

	v = vmax * B / A

	return v





##############################################
######### function to fit v particle #########
##############################################

def vparticlefit(y, w, r, vmax) :
	alpha = r / (2*w)
	s = y / (2*w)
	epsilon1 = s/alpha - 1
	epsilon2 = (1-s)/alpha -1
	Epsilon1 = 1 - alpha/s
	Epsilon2 = 1 - alpha/(1-s)
	L1 = np.log10(Epsilon1)
	L2 = np.log10(Epsilon2)

	si = s - 1/2

	fTxx = 0.954 - 8/15*(L1+L2) - 64/375*(Epsilon1*L1+Epsilon2*L2) + 0.0511 + 0.0190*si**2 - 0.613*si**4 + \
		   alpha * (-0.961 - 4.75*si**2 + 0.881*si**4) + alpha**2 * (3.59 + 2.77*si**2 + 84.4*(s-1/4)**4 )
	cTyx = -1/10*(L1-L2) - 43/250*(epsilon1*L1-epsilon2*L2) + 0.0209*(Epsilon1-Epsilon2) - 0.007*si + 0.037*si**3 + \
		   alpha * (0.097*si - 0.310*si**3) + alpha**2 * (2.71*si - 7.21*si**3)
	cRyy = -2/5*(L1+L2) - 66/125*(epsilon1*L1+epsilon2*L2) + 0.1579*(Epsilon1+Epsilon2-1) - 0.206 - 0.0992*si**2 + \
		   0.323*si**4 + alpha * (-0.101 - 1.70*si**2 + 16.83*si**4) + alpha**2 * (0.307 + 15.4*si**2 + 28.7*si**4 )
	fRxy = 4/3 * cTyx
	fP = 1.014 - 4.0454*si**2 + 0.1580*si**4 + alpha * (1.939 + 0.456*si**2 + 11.07*si**4) + \
		 alpha**2 * (3.53 - 14.34*si**2 + 62.88*si**4) + alpha**3 * (-1.40 + 24.28*si**2 - 212.38*si**4)
	cP = 0.0095*si - 0.0645*si**3 + alpha * (-4.382*si + 2.631*si**3) + alpha**2 * (-0.328*si + 11.323*si**3)

	u = 4 * vmax * (cRyy*fP + fRxy*cP) / (cRyy*fTxx - fRxy*cTyx)

	return u







#####################################
######### function to fit y #########
######################################

def yfit( x, delta, xi, y0, yIni, w, r, h, A, B ):
	### vector x
	xF = np.linspace(x[0], np.max(x)+100e-6, 10000)

	### first method
	# ode
	def dydx( y, xF, xi, delta, y0, w, r, h, A, B ):
		""" we use the velociy profile from a rectangular channel with n from 0 to 1 (included) """
		shear = - np.sinh(pi * (y-w) / h)/A + 1/9 * np.sinh(3 * pi * (y-w) / h)/B
		v = 1-np.cosh(pi * (y-w) / h) / A-1 / 18 * (1-np.cosh(3 * pi * (y-w) / h) / B)
		mig = xi * r**(delta+1) * pi / (h * (y-y0)**delta)
		return mig * shear / v

	# calcul position y
	yOde = sci.odeint(dydx, yIni, xF, args=(xi, delta, y0, w, r, h, A, B))
	yOde = yOde[:, 0]

	### interpolation
	f = sci2.interp1d(xF, yOde)
	y = f(x)

	return y





#####################################
######### function to fit y #########
#####################################

def yfitsimple(x, delta, xi, yIni, w, r) :
		### first method 
			# ode
	def dydx(y, x, xi, delta, w, r) :
		return (xi * (r/y)**(delta+1)) * ((w-y)/(w-y/2))
		
		# calcul position y
	y = sci.odeint(dydx, yIni, x, args=(xi,delta,w,r))
	yShape = np.size(y); y = np.reshape(y,yShape)

	return y




########################################
######### function to fit y(t) #########
########################################

def fityx(x, delta, beta, yIni, w, r):
	### first method
		# ode
	def dydx(y, x, delta, beta, w, r):
		return (1 / (delta+1)) * (beta*r)**(delta+1) * ( (w - y) / ( (w*y-y**2/2) * (y-yIni)**delta ) )

		# calcul position y
	y = sci.odeint(dydx, yIni, x, args=(delta, beta, w, r))
	yShape = np.size(y); y = np.reshape(y, yShape)

	return y





###########################################
######### function to fit a ratio #########
###########################################

def afit( b, delta, xi, a0, aIni, w, r, h):
	### vector x
	bF = np.linspace(b[0], np.max(b)+100e-6, 10000)

	### first method
	# ode
	def dydx( a, bF, xi, delta, a0, w, r, h):
		""" we use the velociy profile from a rectangular channel with n from 0 to 1 (included) """
		shear = - np.sinh(pi*(a-1)/h)/cosh(pi/h) + 1/9 * np.sinh(3*pi*(a-1)/h)/cosh(3*pi/h)
		v = 1 - np.cosh(pi*(a-1)/h)/cosh(pi*(1-a0)/h) - 1/27 * (1-np.cosh(3*pi*(a-1)/h)/cosh(3*pi*(1-a0)/h))
		mig = xi * r**(delta+1) * pi / (h * a**delta)

		return mig * shear / v

	# calcul position y
	aOde = sci.odeint(dydx, aIni, bF, args=(xi, delta, a0, w, r, h))
	aOde = aOde[:, 0]

	### interpolation
	f = sci2.interp1d(bF, aOde)
	a = f(b)

	return a





###########################################
######### function to fit a ratio #########
###########################################
def ratiofitsimple(b, delta, xi, aIni, R) :
		### first method 
			# ode
	def dadb(a, b, xi, delta, R) :
		return xi * (R/a)**(delta+1) * ((1-a)/(1-a/2))
		
		# calcul position y
	a = sci.odeint(dadb, aIni, b, args=(xi, delta, R))
	aShape = np.size(a); a = np.reshape(a,aShape)

	return a





###########################################
######### function to fit a ratio #########
###########################################
def fityollasimple(x, U, delta, y0, yIni, w, r, h):
	### vector x
	xF = np.linspace(x[0], np.max(x)+100e-6, 10000)

	### first method
	# ode
	def dydx( y, xF, u, delta, y0, w, r, h):
		""" we use the velociy profile from a rectangular channel with n from 0 to 1 (included) """
		shear = - np.sinh(pi*(y-w)/h)/cosh(pi*w/h) + 1/9*np.sinh(3*pi*(y-w)/h)/cosh(3*pi*w/h)
		v = 1 - np.cosh(pi*(y-w)/h)/cosh(pi*(w-y0)/h) - 1/18*(1 - np.cosh(3*pi*(y-w)/h)/cosh(3*pi*(w-y0)/h))
		mig = u * r**(delta+1) * pi / (h * y**delta)
		return mig * shear / v

	# calcul position y
	yOde = sci.odeint(dydx, yIni, xF, args=(U, delta, y0, w, r, h))
	yOde = yOde[:, 0]

	### interpolation
	f = sci2.interp1d(xF, yOde)
	y = f(x)


	return y







###########################################
######### function to fit a y(tN) #########
###########################################

def fitytN( tN, alpha, beta, yIni, width):
	### vector x
	tNF = np.linspace(tN[0], np.max(tN)+10, 10000)

	### first method
		# ode
	def dadtN(a, tNF, alpha, beta, width) :
		return beta * (width - a) /(width * a**alpha)

		# calcul position y
	aOde = sci.odeint(dadtN, yIni, tNF, args=(alpha, beta , width))
	aOde = aOde[:,0]

	### interpolation
	f = sci2.interp1d(tNF, aOde)
	a = f(tN)

	return a






#####################################################
######### function to fit a log(y(log(tN))) #########
#####################################################

def fitlogytN( logt, alpha, logBeta):
	### first method
	logy = alpha * logt + logBeta

	return logy







#####################################################
######### function to fit a log(y(log(tN))) #########
#####################################################

def fitdirect( t, alpha, xi, yIni):
	### first method
	y = ( alpha * xi * t + yIni**alpha )**(1/alpha)

	return y



