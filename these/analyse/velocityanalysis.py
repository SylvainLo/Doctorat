import numpy as np

import matplotlib.pyplot as plt

import os

import lmfit as lm

import basefunction
import fuctionforfit







#####################################
######### Velocity Analysis #########
#####################################
def velocityanalysis(nFile, nameFile, baseValue, subDirectoryVideo) :
	### Loading data
	os.chdir(subDirectoryVideo)
	data = np.loadtxt(nameFile[nFile])
	print(nameFile[nFile])


	### Variable
		# Variable for loop
	nbrTrack = int(data[-1,0])
	c = int(0)
	
		# Variable for analysis
	nbrParticle = int(data[-1,1])
	velocity = np.zeros((nbrParticle, 3), dtype=np.float)


	### Measure velocity for each track
	for i in range(nbrTrack) :
		# Creation of the vector from one track
		vecTrack = data[ np.all([data[:,0] == i], axis=0) ]
		(rowF, col) = np.shape(vecTrack)
		l = c + rowF - 1

		# Velocity and y position for one track
		if rowF > 3 :
			velocity[c:l, 0] = (vecTrack[:-1, 5] + vecTrack[1:, 5]) / 2  # distance wall/Rbc
			velocity[c:l, 1] = np.abs((vecTrack[:-1, 3] - vecTrack[1:, 3] )) * baseValue.distance[nFile,4]  # velocity
			velocity[c:l, 2] = (vecTrack[:-1, 5] + vecTrack[:-1,6] + vecTrack[1:, 5] + vecTrack[1:,6]) / 4 # half-width
			c = c + rowF - 1


	### Reduction size vector velocity (take only the one which are different form 0)
		# Going from pix to m
	velocity = velocity * (baseValue.distance[nFile,3]/baseValue.distance[nFile,4])

		# Reducing the size of the vector
	velocity = np.compress(velocity[:, 0] != 0, velocity, axis=0)
	

	### Save local data
	print(velocity[:,2])
	velocityLocalFile = 'velocityLocalFile_%03d.txt'%(nFile+1)
	np.savetxt(velocityLocalFile, velocity, fmt='%.4e %.4e %.4e', delimiter='    ')

	return







#########################################
######### Construction all data #########
#########################################
def constructionalldata(subDirectoryVideo, width) :
	### Search all data
	(nameDataLocal, sizeDataLocal) = basefunction.filestoanalyse(subDirectoryVideo, 'velocityLocalFile*')

	### Creation new vector
		# creating a new vector
	datTotal = np.array([])

		# filling this vector
	for nFilm in range(sizeDataLocal) :
		datLocal = np.loadtxt(nameDataLocal[nFilm])
		if nFilm == 0 :
			datTotal = datLocal
		else :
			datTotal = np.concatenate((datTotal, datLocal), axis=0)

	### Reorganize data in function of y
	datTotal = datTotal[datTotal[:,0] < datTotal[:,2], :]
	vectorSort = np.argsort(datTotal[:,0])

		# first column reorganisation
	yNoOrder = datTotal[:,0]
	yOrder = yNoOrder[vectorSort]
	datTotal[:,0] = yOrder

		# second column reorganisation
	vNoOrder = datTotal[:,1]
	vOrder = vNoOrder[vectorSort]
	datTotal[:,1] = vOrder

		# third column reorganisation
	wNoOrder = datTotal[:,2]
	wOrder = wNoOrder[vectorSort]
	datTotal[:,2] = wOrder

	return datTotal






################################
######### Fit velocity #########
################################
def fitvelocity(velocity, ax, ax2, baseValue) :
### Variable ###
	radius = 1.00e-6
	width = np.mean(velocity[:,2])
	yFit = np.linspace(0, width, 500)



### Fit rectangular channel ###
	### Fit of the position y
	def vdataset(params, position) :
		V0 = params['V0'].value
		h = params['height'].value
		w = params['width'].value
		n = params['n'].value
		y0 = params['y0'].value

		w = w - y0
		y = position - y0
		return fuctionforfit.vrectangularfit(y, h, w, V0, n)

	### residual function
	def objective(params, position, data) :
		residual = (data - vdataset(params, position))
		return residual.flatten()

	### Calculate delta, xi et yIni optimal with the method differential_evolution
		# Creating parameter
	fitParam0 = lm.Parameters()
	fitParam0.add( 'V0', np.max(velocity[:,1]), min=0, max=np.max(velocity[:,1]))
	fitParam0.add( 'y0', radius, min=0, max=np.mean(velocity[:,2]))
	fitParam0.add( 'height', baseValue.distance[0,9], vary=None)
	fitParam0.add( 'width', width, vary=None)
	fitParam0.add( 'n', 21, vary=None)

		# Calcul fit
	result = lm.minimize(objective, fitParam0, method='differential_evolution', args=(velocity[:,0], velocity[:,1]), nan_policy='omit')

		# Show result
	print('------------ {} ------------'.format(baseValue.name_file))
	lm.report_fit(result.params)

		# Save result
	Vmax = np.array([result.params['V0'], result.params['y0']])
	np.savetxt('V0.txt', Vmax)


	### Plot
		# plot 1
	vFromFit = vdataset(result.params, yFit)
	vFlow = 2 * Vmax[0] / (width-Vmax[1])**2 * ( (yFit-Vmax[1])*(width-Vmax[1]) - (yFit-Vmax[1])**2/2)
	ax.plot( yFit, vFromFit, 'r-',
			 label = ('curve velocity rectangular channel : V0={:.2E}, y0={:.2}'.format(Vmax[0], Vmax[1]*1e3)) )
	ax.plot( yFit, vFlow, 'g:', label='parabolic curve')
	ax.legend()

		# plot2
	shear = ( 2 * result.params['V0'] / velocity[:,2]**2 ) * ( velocity[:,2] - velocity[:,0] )
	ax2.plot( shear, velocity[:,1], ls='None', marker='o', markersize=1, alpha=0.1, color='b')



### Fit particle velocity ###
	### Fit of the position y
	def vdataset2( params, position ):
		V0 = params['V0'].value
		w = params['width'].value
		r = params['radius'].value

		y = position
		return fuctionforfit.vparticlefit(y, w, r, V0)


	### residual function
	def objective2( params, position, data ):
		residual = (data-vdataset2(params, position))
		return residual.flatten()


	### Calculate Vmax
		# Creating parameter
	fitParam2 = lm.Parameters()
	fitParam2.add('V0', np.max(velocity[:, 1]), min=0, max=np.max(velocity[:, 1]))
	fitParam2.add('radius', 1e-7, min=1e-9, max=velocity[:,0].min())
	fitParam2.add('width', np.mean(velocity[:, 2]), vary=None)

		# Calcul fit
	yParticle = np.compress(radius<yFit, yFit)
	yParticle = np.compress(yParticle<2*width-radius, yParticle)

	result2 = lm.minimize(objective2, fitParam2, method='differential_evolution',
						  args=(velocity[velocity[:, 0]>radius, 0], velocity[velocity[:, 0]>radius, 1]), nan_policy='omit')

	# Show result
	print('------------ {} ------------'.format(baseValue.name_file))
	lm.report_fit(result2.params)


	### Plot
		# plot 1
	vFromFit = vdataset2(result2.params, yParticle)
	ax.plot(yParticle, vFromFit, 'k-', label=('curve particle velocity : V0={:.2E} and radius={:.2E}'.
											  format(result2.params['V0'].value, result2.params['radius'].value)) )
	ax.set_ylim([0, velocity[:, 1].max()])
	ax.legend()

	return ax, ax2






#################################
######### Plot velocity #########
#################################
def plotvelocity(nFiles, nameDirectory, nAllFile, baseValue) :
	### change file
	subDirectory = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectory)


	### Creating global Data
		# Width
	test = np.load('totalLocalAnalysisData.npy')
	width = np.mean(test[:,5])

		# Vector all data
	velocity = constructionalldata(subDirectory, width)
			

	### Plot data
		# plot for fit
	fig = plt.figure('velocity of {}'.format(baseValue.name_file))
	ax1 = fig.add_subplot(111)
	ax1.plot(velocity[:,0], velocity[:,1], ls='None', marker='o', markersize=1, alpha=0.1, color='b')

		# plot shear/velocity (y)
	fig2 = plt.figure(r'v\\\gamma (y) ')
	ax2 = fig2.add_subplot(111)


	### Fit
	ax1, ax2= fitvelocity(velocity, ax1, ax2, baseValue)


	### Save fig
	fig.savefig('V(y).pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape')
	fig2.savefig('V(shear).pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape')

	return


