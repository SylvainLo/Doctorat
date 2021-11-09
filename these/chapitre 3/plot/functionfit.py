import numpy as np

import scipy.integrate as sci
import scipy.interpolate as sci2

from math import pi, cosh, sin, sinh



#####################################
######### function to fit v #########
#####################################

def vrectangularfit(y, h, vmax, n) :
	A = 0 ; B = 0

	for i in range(n) :
		AB1 = (-1)**i / (2*i+1)**3
		AB2 = cosh((2*i+1)*pi/h)
		A = A + AB1 * (1 - 1/AB2)

		B3 = np.cosh((2*i+1)*pi*(y-1)/h)
		B = B + AB1 * (1 - B3/AB2)

	v = vmax * B / A

	return v





##############################################
######### function to fit v particle #########
##############################################

def vparticlefit(y, r, vmax) :
	alpha = r / 2
	s = y / 2
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

	u = vmax * (cRyy*fP + fRxy*cP) / (cRyy*fTxx - fRxy*cTyx)

	return u





##################################################################
######### function to fit v integral rectangular channel #########
##################################################################

def intv3dflow(y, h, vMax, r, nLim) :
	A = 0 ; B = 0

	for ii in range(1,nLim+1,2) :
		c1 = (-1)**(int((ii-1) / 2)) / ii**3
		c2 = ii * pi / h

		a1 = cosh(c2)

		A = A + c1 * (1 - 1/a1)
		B = B + c1/ii * (r - ( np.cosh(c2*(y-1))*sinh(c2*r) )/( c2*a1 )) *sin(c2*r)

	v = (vMax / r**2) *  (B / A)

	return v




##################################################################
######### function to fit v integral rectangular channel #########
##################################################################

def intv3dflowyz(y, h, r, nLim, vMax) :
	A = float(0) ; B = float(0)

	for ii in range(1, nLim+1, 2) :
		A1 = (-1)**((ii-1)/2) / ii**3
		A2 = ii * pi / h

		A = A + A1*(1 - 1/cosh(A2))
		B = B + A1/A2 * (r - ( (sinh(A2*r)*np.cosh(A2*(y-1))) / (A2*cosh(A2)) ) ) * sin(A2*r)

	v = vMax/r**2 * B/A

	return v




#####################################
######### parabolic channel #########
#####################################
def intparabolic(y,V0,r) :
	return 2*V0 * (y - y**2/2 - r/6)





###########################################
######### function to fit a ratio #########
###########################################

def afit( b, delta, xi, rExp, aIni, h, rThe, n):
	### vector x
	bF = np.linspace(b[0], np.max(b)+1, 10000)

	### first method
		# ode
	def dadb( a, bF, delta, xi, rExp, h, rThe, n):
		""" we use the velociy profile from a rectangular channel with limn = n (included) """
		A = 0 ; B = 0

		for ii in range(1,n+1,2) :
			c1 = (-1)**(int((ii-1)/2)) / ii**3
			c2 = ii * pi / h

			a1 = cosh(c2)

			A = A + ii*c1 * ( np.sinh(c2*(a-1)) / a1 )
			B = B + c1/ii * ( rExp - (np.cosh(c2*(a-1))*sinh(c2*rExp))/(c2*a1) ) * sin(c2*rExp)

		cte = xi * rExp**2 * rThe**(delta+1) * pi**2 / (a**delta * h**2)
		va = - cte * (A / B)
		return va

		# calcul position y
	aOde = sci.odeint(dadb, aIni, bF, args=(delta, xi, rExp, h, rThe, n))
	aOde = aOde[:, 0]

	### interpolation
	f = sci2.interp1d(bF, aOde)
	a = f(b)

	return a




###############################################
######### function to fit a ratio Olla#########
###############################################

def afitU( b, delta, U, rExp, aIni, h, n):
	### vector x
	bF = np.linspace(b[0], np.max(b)+1, 10000)

	### first method
		# ode
	def dadb( a, bF, delta, U, rExp, h, n):
		""" we use the velociy profile from a rectangular channel with limn = n (included) """
		A= 0 ; B= 0
		for ii in range(1,n+1,2) :
			AB1 = (-1)**(int((ii-1)/2)) / ii**3
			AB2 = cosh(ii * pi / h)

			A = A - ii * AB1 * (np.sinh(ii*pi*(a-1)/h) / AB2)
			B = B + AB1/ii * (2*rExp + (np.sinh(ii*pi*(a-rExp-1)/h) - np.sinh(ii*pi*(a+rExp-1)/h))/(ii*pi/h*AB2)) * sin(ii*pi*rExp/h)

		cte = U / a**delta
		va = cte * (A / B)
		return va

		# calcul position y
	aOde = sci.odeint(dadb, aIni, bF, args=(delta, U, rExp, h, n))
	aOde = aOde[:, 0]

	### interpolation
	f = sci2.interp1d(bF, aOde)
	a = f(b)

	return a





###################################################
######### function to fit a ratio Gwennou #########
###################################################

def afitG( b, delta, xi, rExp, aIni, h, rThe, n):
	### vector x
	bF = np.linspace(b[0], np.max(b)+1, 10000)

	### first method
		# ode
	def dadb( a, bF, delta, xi, rExp, h, rThe, n):
		""" we use the velociy profile from a rectangular channel with limn = n (included) """
		A = 0 ; B = 0

		for ii in range(1,n+1,2) :
			c1 = (-1)**(int((ii-1)/2)) / ii**3
			c2 = ii * pi / h

			a1 = cosh(c2)

			A = A + ii*c1 * ( np.sinh(c2*(a-1)) / a1 )
			B = B + c1/ii * ( rExp - (np.cosh(c2*(a-1))*sinh(c2*rExp))/(c2*a1) ) * sin(c2*rExp)

		cte = xi * rExp**2 * rThe**(delta+1) * pi**2 / ((a-rThe)**delta * h**2)
		va = - cte * (A / B)
		return va

		# calcul position y
	aOde = sci.odeint(dadb, aIni, bF, args=(delta, xi, rExp, h, rThe, n))
	aOde = aOde[:, 0]

	### interpolation
	f = sci2.interp1d(bF, aOde)
	a = f(b)

	return a