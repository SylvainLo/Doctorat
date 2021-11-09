import os

import numpy as np			
import scipy.integrate as sci

import math

import matplotlib.pyplot as plt	

import basefunction




##################################
######### Local analysis #########
##################################
def localanalysis(nFiles, nameDirectory, nAllFile, baseValue) :
	### Search data ###
		### Change directory
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)

		### List all fill containing data
	(nameDataTrack, sizeDataTrack) = basefunction.filestoanalyse(subDirectoryVideo, '*trackDataFile*')


	### Preparation plot ###
	fig1 = plt.figure( 'distance and ratio all point {}'.format(baseValue.name_file) )
	ax11 = fig1.add_subplot(211)
	ax11.set_xlabel('x') ; ax11.set_ylabel('distance Wall/RBC')
	ax12 = fig1.add_subplot(212)
	ax12.set_xlabel('x') ; ax12.set_ylabel('ratio')

	fig2 = plt.figure( 'width channel, angle and radius for {}'.format(baseValue.name_file) )
	ax21 = fig2.add_subplot(311)
	ax21.set_xlabel('x') ; ax21.set_ylabel('width')
	ax22 = fig2.add_subplot(312)
	ax22.set_xlabel('x') ; ax22.set_ylabel('small radius (red), bid radius (blue)')
	ax23 = fig2.add_subplot(313)
	ax23.set_xlabel('x') ; ax23.set_ylabel('angle')
	
	fig3 = plt.figure( 'distance Wall/Wall for {}'.format(baseValue.name_file) )
	ax31 = fig3.add_subplot(211)
	ax31.set_xlabel('x') ; ax31.set_ylabel('small distance Wall/Wall')
	ax32 = fig3.add_subplot(212)
	ax32.set_xlabel('x') ; ax32.set_ylabel('big distance Wall/Wall')


	### Lecture data ###
	for nFileTrack in range(sizeDataTrack) :
		data = np.loadtxt(nameDataTrack[nFileTrack]) ; print(nameDataTrack[nFileTrack])

		### Introduction variable
			# Variable for loop
		(row, column) = np.shape(data)
		nbrTrack = int(np.max(data[:,0]))
		c = 0

			# Variable for analysis
		vecAnalysis = np.zeros( (row,12), dtype=float )

			# Correction x by sign velocity
		if baseValue.distance[nFileTrack,7] < 0 :
			data[:,3] =  baseValue.distance[nFileTrack,1] + (baseValue.distance[nFileTrack,8] - data[:,3])
		else :
			data[:,3] = baseValue.distance[nFileTrack,1] + data[:,3]
	
		### Local Analysis
		for i in range(int(nbrTrack)) : 
			mask = np.argwhere(data[:,0]==i)
			vecTrack = data[mask[:,0],:]
			lim = np.shape(vecTrack)[0]

			if lim > 3 :
				l = c + lim

				vecAnalysis[c:l,0] = vecTrack[:,0]  # number track
				vecAnalysis[c:l,1] = vecTrack[:,3]  # x
				vecAnalysis[c:l,2] = vecTrack[:,4]  # y
			
				vecAnalysis[c:l,3] = vecTrack[:,5]  # small distance
				vecAnalysis[c:l,4] = vecTrack[:,6]  # big distance
			
				vecAnalysis[c:l,5] = (vecAnalysis[c:l,3] + vecAnalysis[c:l,4]) / 2 # size channel
				vecAnalysis[c:l,6] = vecAnalysis[c:l,3] / vecAnalysis[c:l,5] # ratio

				vecAnalysis[c:l,7] = vecTrack[:,7]  # small distance wall wall
				vecAnalysis[c:l, 8] = vecTrack[:,8]  # big distance wall wall

				vecAnalysis[c:l,9] = vecTrack[:,9]  # big diameter
				vecAnalysis[c:l,10] = vecTrack[:,10]  # small diameter
				vecAnalysis[c:l,11] = vecTrack[:,11]  # angle

				ax11.plot(vecAnalysis[c:l,1], vecAnalysis[c:l,3], linewidth=0.01, markersize=1)
				ax12.plot(vecAnalysis[c:l,1], vecAnalysis[c:l,6], linewidth=0.01, markersize=1)

				ax31.plot(vecAnalysis[c:l, 1], vecAnalysis[c:l, 7], linewidth=0.01, markersize=1)
				ax32.plot(vecAnalysis[c:l, 1], vecAnalysis[c:l, 8], linewidth=0.01, markersize=1)
			
				c += lim
	

		### Last local Analysis
			# reduction local Analysis
		vecAnalysis = np.compress(vecAnalysis[:,1]!=0, vecAnalysis, axis=0)

			# passage pixel/um
		a = (baseValue.distance[nFileTrack,3] / baseValue.distance[nFileTrack,2])
		vecAnalysis[:,1:6] = vecAnalysis[:,1:6] * a # utilisation de : nécessite d'ajouter un colonne, calcul sur [:,1:4]
		vecAnalysis[:,7:11] = vecAnalysis[:,7:11] * a

			# plot
		ax21.plot(vecAnalysis[:,1], vecAnalysis[:,5], 'o', markersize=0.1)
		ax22.plot(vecAnalysis[:,1], vecAnalysis[:,9], 'o', color='red', markersize=0.1)
		ax22.plot(vecAnalysis[:,1], vecAnalysis[:,10], 'o', color='blue', markersize=0.1)
		ax23.plot(vecAnalysis[:,1], vecAnalysis[:,11], 'o', markersize=0.1)

			# Saving local analysis
		localAnalysisFile = 'localAnalysisData_%03d.txt'%(nFileTrack+1)
		np.savetxt(localAnalysisFile, vecAnalysis, fmt = '%.0d %.4e %.4e %.4e %.4e %.4e %.3f %.4e %.4e %.4e %.4e %.3f' ,
				   delimiter='	', header="nTrack, x , y , small d, big D, halfWidth, ratio size, d Wall//Wall, D Wall/Wall, big radius, small radius, angle")


	### Construction all data ###
	constructionalldata(subDirectoryVideo)

	### Save figure ###
	fig1.savefig( 'distance and ratio all point {}.pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape' )
	fig2.savefig( 'W, angle and radius {}.pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape' )
	fig3.savefig( 'distance wall wall for {}.pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape' )

	return









#########################################
######### Construction all data #########
#########################################
def constructionalldata(subDirectoryVideo) :
	### Search all data
	(nameDataLocal, sizeDataLocal) = basefunction.filestoanalyse(subDirectoryVideo, 'localAnalysisData*')

	### Creation new vector
		# Creating a new vector
	data = np.array([])

		# Fill this new vector
	for nFilm in range(sizeDataLocal) :
		datLocal = np.loadtxt(nameDataLocal[nFilm]) ; print(nameDataLocal[nFilm])

		if nFilm == 0 :
			data = datLocal
		else :
			data = np.concatenate((data, datLocal), axis=0)

	### Saving data
	totalLocalAnalysisFile = 'totalLocalAnalysisData.npy'
	np.save(totalLocalAnalysisFile, data)

	return










###################################
######### Global analysis #########
###################################
def globalanalysis(nameDirectory, nFiles, nAllFile, baseValue) :
	### Change of directory ###
	subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
	os.chdir(subDirectoryVideo)

	### Lecture data ###
	data = np.load('totalLocalAnalysisData.npy')
	V = np.loadtxt('V0.txt')

	### mean data and error global ###
		### Vector separation
	vecSep = np.arange(0, np.max(data[:,1])+baseValue.delta_x, baseValue.delta_x)
	nBins = np.size(vecSep)

		### Result vector
	datGlobal = np.zeros( (nBins, 18) )

		### Complete the result vector
	for i in range(nBins-1) :
		vecAnalysis = data[ np.all([data[:,1]>vecSep[i], data[:,1]<vecSep[i+1]], axis=0) ]
		(r,c) = np.shape(vecAnalysis)

		if r > 1 :
			datGlobal[i,0] = np.mean(vecAnalysis[:,1])  # x
			datGlobal[i,1] = np.mean(vecAnalysis[:,5])  # size channel

			datGlobal[i,2] = np.mean(vecAnalysis[:,3])  # mean d
			datGlobal[i,3] = 2.576 * np.std(vecAnalysis[:, 3]) / math.sqrt(r)  # incertitude on mean d

			datGlobal[i,4] = np.mean(vecAnalysis[:,6])  # mean ratio
			datGlobal[i,5] = 2.576 * np.std(vecAnalysis[:, 6]) / math.sqrt(r)  # incertitude on mean ratio

			datGlobal[i,6] = np.mean(vecAnalysis[:,7])  # mean d wall/wall
			datGlobal[i,7] = 2.576 * np.std(vecAnalysis[:, 7]) / math.sqrt(r)

			datGlobal[i,8] = np.mean(vecAnalysis[:,8])  # mean D wall/wall
			datGlobal[i,9] = 2.576 * np.std(vecAnalysis[:, 8]) / math.sqrt(r)

			datGlobal[i,10] = np.mean(vecAnalysis[:,9])  # big radius
			datGlobal[i,11] = 2.576 * np.std(vecAnalysis[:, 9]) / math.sqrt(r)

			datGlobal[i,12] = np.mean(vecAnalysis[:,10])  # small radius
			datGlobal[i,13] = 2.576 * np.std(vecAnalysis[:, 10]) / math.sqrt(r)

			datGlobal[i,14] = np.mean(vecAnalysis[:,11])  # angle
			datGlobal[i,15] = 2.576 * np.std(vecAnalysis[:, 11]) / math.sqrt(r)


	datGlobal = np.compress(datGlobal[:, 2] != 0, datGlobal, axis=0)


	### Fill global time with mean time ###
		### Real time
			# different variable
	width = np.mean(datGlobal[:,1])
	y = datGlobal[:,2]
	x = datGlobal[:,0]
	Vx = V[0] * (1 - V[1] * ((y - width) / width) ** 2)

			# Integral on experimental data
	Y = 1/Vx
	time = sci.cumtrapz(Y, x, initial=0)
	datGlobal[:,16] = time

		### Normalize time
	shear = ((2 * V[0]) / width**2) * (width-y)
	timeN = sci.cumtrapz(shear, time, initial=0)
	datGlobal[:,17] = timeN


	### Saving data ###
	globalAnalysisFile = 'globalAnalysisData.txt'
	np.savetxt(globalAnalysisFile, datGlobal, fmt = '%.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e',
			   delimiter='	', header="x, width, d, error, ratio, error, d wall//wall, error, R, error, r, error, angle, error, time, timeN")
	print('data for global analysis have been saved')


	### Plot ###
		# fig 1 : distance and ratio for different variable
	fig1 = plt.figure('distance from wall (and error) {}'.format(baseValue.name_file))
	ax11 = fig1.add_subplot(321)
	ax11.errorbar(datGlobal[:,0], datGlobal[:,2], yerr=datGlobal[:,3], fmt='o', markersize=1, elinewidth=0.1)
	ax11.set_xlabel('x') ; ax11.set_ylabel('distance Wall/RBC')

	ax12 = fig1.add_subplot(322)
	ax12.errorbar(datGlobal[:,0], datGlobal[:, 4]-datGlobal[0, 4], yerr=datGlobal[:,5], fmt='o', markersize=1, elinewidth=0.1)
	ax12.set_xscale("log", nonposx='clip') ; ax12.set_yscale("log", nonposy='clip')
	ax12.set_xlabel('x') ; ax12.set_ylabel('ratio')

	ax13 = fig1.add_subplot(323)
	ax13.errorbar(datGlobal[:, 16], datGlobal[:, 2], yerr=datGlobal[:, 3], fmt='o', markersize=1, elinewidth=0.1)
	ax13.set_xlabel('t') ; ax13.set_ylabel('distance Wall/RBC')

	ax14 = fig1.add_subplot(324)
	ax14.errorbar(datGlobal[:, 16], datGlobal[:, 4]-datGlobal[0, 4], yerr=datGlobal[:, 5], fmt='o', markersize=1, elinewidth=0.1)
	ax14.set_xscale("log", nonposx='clip') ; ax14.set_yscale("log", nonposy='clip')
	ax14.set_xlabel('t') ; ax14.set_ylabel('ratio')

	ax15 = fig1.add_subplot(325)
	ax15.errorbar(datGlobal[:, 17], datGlobal[:, 2], yerr=datGlobal[:, 3], fmt='o', markersize=1, elinewidth=0.1)
	ax15.set_xlabel('tN') ; ax15.set_ylabel('distance Wall/RBC')

	ax16 = fig1.add_subplot(326)
	ax16.errorbar(datGlobal[:, 17], datGlobal[:, 4], yerr=datGlobal[:, 5], fmt='o', markersize=1, elinewidth=0.1)
	#ax16.set_xscale("log", nonposx='clip') ; ax16.set_yscale("log", nonposy='clip')
	ax16.set_xlabel('tN') ; ax16.set_ylabel('ratio')

		# fig 2 : distance center of mass and wall and wall/wall
	fig2 = plt.figure('distance from wall (ecart type/Vn) of {}'.format(baseValue.name_file))
	ax21 = fig2.add_subplot(111)
	ax21.errorbar(datGlobal[:,0], datGlobal[:,2], yerr=datGlobal[:,3], fmt='o', markersize=1, elinewidth=0.1, color='red')
	ax21.errorbar(datGlobal[:,0], datGlobal[:,6], yerr=datGlobal[:,7], fmt='o', markersize=1, elinewidth=0.1, color='blue')
	ax21.errorbar(datGlobal[:,0], datGlobal[:,8], yerr=datGlobal[:,9], fmt='o', markersize=1, elinewidth=0.1, color='navy')
	ax21.set_xlabel('x') ; ax21.set_ylabel('distance')

		# fig 3 : angle and radius
	fig3 = plt.figure('angle and radius RBC of {}'.format(baseValue.name_file))
	ax31 = fig3.add_subplot(211)
	ax31.errorbar(datGlobal[:,0], datGlobal[:,10], yerr=datGlobal[:,11], fmt='o', markersize=1, color='red')
	ax31.errorbar(datGlobal[:,0], datGlobal[:,12], yerr=datGlobal[:,13], fmt='o', markersize=1, color='blue')
	ax31.set_xlabel('x') ; ax31.set_ylabel('radius')

	ax32 = fig3.add_subplot(212)
	ax32.errorbar(datGlobal[:,0], datGlobal[:,14], yerr=datGlobal[:,15], fmt='o', markersize=1)
	ax32.set_xlabel('x') ; ax32.set_ylabel('angle')


	### Save figure ###
	fig1.savefig('distance and ratio (+error).pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape')
	fig2.savefig('distance center and wall wall and wall (+error).pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape')
	fig3.savefig('angle and radius (+error).pdf'.format(baseValue.name_file), frameon=False, bbox_inch='tight', orientation='landscape')

	return()
