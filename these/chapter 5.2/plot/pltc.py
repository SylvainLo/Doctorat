import numpy as np
import matplotlib.pyplot as plt
import math

import os

import fitingfunction
import basefunction as bf




###########################################
#########  #########
###########################################
def plotconcentration(fig, dir, cf, sbf, lsf) :
	#################
	### Plot
	ax1 = fig.add_axes( [0.10, 0.15, 0.80, 0.80] ) #[left,bot,width,height]
	ax1.set_xlabel( r'$x$ (mm)') ; ax1.set_ylabel( r'$\Delta\tau$ (s)' )

	# ax with surface
	ax2 = fig.add_axes( [0.11, 0.65, 0.30, 0.29] ) #[left,bot,width,height]
	ax2.set_xlabel( r'$\Phi$' ) ; ax2.set_ylabel( r'$\xi$' )
	ax2.yaxis.set_label_position( "right" ) ; ax2.yaxis.tick_right( )


	#################
	### Detection Subfile of interest
	dirC = os.path.join( dir, 'concentration/' )
	os.chdir(dirC)
	(nFile, sFile) = bf.filestoanalyse( dirC, '*IMAGE*' )
	title = np.loadtxt('val.txt')

	cn = int( 0 )

	### Detection movie inside each file
	for n in nFile :
		# directory
		dirSub = os.path.join( dirC, n )
		os.chdir( dirSub )
		(nSubFile, sSubFile) = bf.filestoanalyse( dirSub, '*%*' )

		for nn in nSubFile :
		# sub directory
			nDirSub2 = os.path.join( dirSub, nn )
			os.chdir( nDirSub2 )
			print( '\n \n -----------     -------------' )
			print( nDirSub2 )

		# constant value over the movie
			val = np.loadtxt('experimental_value.txt', skiprows=1)
			xExp = val[:,1]*1e-6
			HExp = val[0,2]*1e-6 / 2
			R0 = 2.8e-6

		# load data
			# velocity
			v = np.loadtxt( 'velocity.txt' )
			v0mean = np.loadtxt( 'v0mean.txt' )
			v1mean = np.loadtxt( 'v1mean.txt' )

			# data
			tau = np.loadtxt( 'tau.txt' )
			deltaTau = np.loadtxt( 'deltaTau.txt')
			L = np.loadtxt('L.txt')


	### Fitting result
		# z initial
			z0 = HExp - math.sqrt( HExp**2 * (1-v1mean[0]/v[0]) - R0**2/3 )

		# result to fit
			result = fitingfunction.fitingresult(L, val, z0, v)

		# mean velocity of late RBC from fit
			v1meantheo = fitingfunction.zfit2(xExp, result, v)

		# deltaTau
			deltaTauThe = xExp * (1/v1meantheo - 1/v0mean)


	### Plot result
			ax1.plot(xExp*1000, deltaTau, color=cf[cn], linestyle=None, linewidth=0, marker=sbf[cn], label='\Phi = {} %'.format(title[cn]))
			ax1.plot(xExp*1000, deltaTauThe, color=cf[cn], ls=lsf[cn], linewidth=1)

			ax2.plot(title[cn], result['xi'].value, color=cf[cn], marker=sbf[cn])

	### counter
			cn += 1

	return fig



###########################################
#########  #########
###########################################
def plotlambda(fig, dir, cf, sbf, lsf) :
	#################
	### Plot
	ax1 = fig.add_axes( [0.10, 0.15, 0.85, 0.80] ) #[left,bot,width,height]
	ax1.set_xlabel( r'$x$ (mm)' ) ; ax1.set_ylabel( r'$\Delta\tau$ (s)' )
	ax1.set_ylim( -0.07, 1.75 )


	# ax with surface
	ax2 = fig.add_axes( [0.11, 0.65, 0.30, 0.29] ) #[left,bot,width,height]
	ax2.set_xlabel( r'$\lambda$' ) ; ax2.set_ylabel( r'$\xi$' )
	ax2.yaxis.set_label_position( "right" ) ; ax2.yaxis.tick_right( )
	#ax2.set_xlim( -0.1, 2.3 )


	#################
	### Detection Subfile of interest
	dirC = os.path.join( dir, 'deformability/' )
	os.chdir(dirC)
	(nFile, sFile) = bf.filestoanalyse( dirC, '*IMAGE*' )
	titCa = np.loadtxt('val_Ca.txt')
	titLambda = np.loadtxt( 'val_Lambda.txt' )

	cn = int( 0 )

	### Detection movie inside each file
	for n in nFile :
		# directory
		dirSub = os.path.join( dirC, n )
		os.chdir( dirSub )
		(nSubFile, sSubFile) = bf.filestoanalyse( dirSub, '*%*' )

		for nn in nSubFile :
		# sub directory
			nDirSub2 = os.path.join( dirSub, nn )
			os.chdir( nDirSub2 )
			print( '-----------     -------------' )
			print( nDirSub2 )

		# constant value over the movie
			val = np.loadtxt('experimental_value.txt', skiprows=1)
			xExp = val[:,1]*1e-6
			HExp = val[0,2]*1e-6 / 2
			R0 = 2.8e-6

		# load data
			# velocity
			v = np.loadtxt( 'velocity.txt' )
			v0mean = np.loadtxt( 'v0mean.txt' )
			v1mean = np.loadtxt( 'v1mean.txt' )

			# data
			tau = np.loadtxt( 'tau.txt' )
			deltaTau = np.loadtxt( 'deltaTau.txt')
			L = np.loadtxt('L.txt')


	### Fitting result
		# z initial
			z0 = HExp - math.sqrt( HExp**2 * (1-v1mean[0]/v[0]) - R0**2/3 )

		# result to fit
			result = fitingfunction.fitingresult(L, val, z0, v)

		# mean velocity of late RBC from fit
			v1meantheo = fitingfunction.zfit2(xExp, result, v)

		# deltaTau
			deltaTauThe = xExp * (1/v1meantheo - 1/v0mean)


	### Plot result
			ax1.plot(xExp*1000, deltaTau, color=cf[cn], linestyle=None, linewidth=0, marker=sbf[cn])
			ax1.plot(xExp*1000, deltaTauThe, color=cf[cn], ls=lsf[cn], linewidth=1)

			ax2.plot(titLambda[cn], result['xi'].value, color=cf[cn], marker=sbf[cn])

	### counter
			cn += 1

	return fig




###########################################
#########  #########
###########################################
def plotdensity(fig, dir, cf, sbf, lsf) :
	#################
	### Plot
	ax1 = fig.add_axes( [0.10, 0.15, 0.80, 0.80] ) #[left,bot,width,height]
	ax1.set_xlabel( r'$x$ (mm)' ) ; ax1.set_ylabel( r'$\Delta\tau$ (s)' )



	#################
	### Detection Subfile of interest
	dirC = os.path.join( dir, 'density/' )
	os.chdir(dirC)
	(nFile, sFile) = bf.filestoanalyse( dirC, '*IMAGE*' )

	cn = int( 0 )

	### Detection movie inside each file
	for n in nFile :
		# directory
		dirSub = os.path.join( dirC, n )
		os.chdir( dirSub )
		(nSubFile, sSubFile) = bf.filestoanalyse( dirSub, '*%*' )

		for nn in nSubFile :
		# sub directory
			nDirSub2 = os.path.join( dirSub, nn )
			os.chdir( nDirSub2 )
			print( '-----------     -------------' )
			print( nDirSub2 )

		# constant value over the movie
			val = np.loadtxt('experimental_value.txt', skiprows=1)
			xExp = val[:,1]*1e-6
			HExp = val[0,2]*1e-6 / 2
			R0 = 2.8e-6

		# load data
			# velocity
			v = np.loadtxt( 'velocity.txt' )
			v0mean = np.loadtxt( 'v0mean.txt' )
			v1mean = np.loadtxt( 'v1mean.txt' )

			# data
			tau = np.loadtxt( 'tau.txt' )
			deltaTau = np.loadtxt( 'deltaTau.txt')
			L = np.loadtxt('L.txt')


	### Fitting result
		# z initial
			z0 = HExp - math.sqrt( HExp**2 * (1-v1mean[0]/v[0]) - R0**2/3 )

		# result to fit
			result = fitingfunction.fitingresult(L, val, z0, v)

		# mean velocity of late RBC from fit
			v1meantheo = fitingfunction.zfit2(xExp, result, v)

		# deltaTau
			deltaTauThe = xExp * (1/v1meantheo - 1/v0mean)


	### Plot result
			ax1.plot(xExp*1000, deltaTau, color=cf[cn], linestyle=None, linewidth=0, marker=sbf[cn])
			ax1.plot(xExp*1000, deltaTauThe, color=cf[cn], ls=lsf[cn], linewidth=1)

	### counter
			cn += 1

	return fig