import numpy as np 				#print('numpy imported')
import os 						#print('os imported')
import matplotlib.pyplot as plt	#print('matplotlib imported')
import joblib					#print('joblib imported')
import multiprocessing			#print('multiprocessing imported')


import basefunction
import reperageglobule
import globuletracking
import velocityanalysis
import analysis
import parameterdispersion
import fitglobal
import fittotal
import fitpoint

print('modules have been charged')




##############################
######### Main code ##########
##############################


if __name__ == '__main__':
################################################################################################
	### Change directory ###
	nameDirectory = 'F:/test'
	#nameDirectory = input('Indicate the place of your data (E:/.../) : ') ;
	while os.path.isdir(nameDirectory) is False :
		print('   Error : your directory doesn t exist, please indicate a correct path')
		nameDirectory = input('Indicate the place of your data (E:/.../) : ')

	os.chdir(nameDirectory) ; print('   your directory exist !')

	### Search Sub directory ###
	(nAllFile,sizeAllFile) = basefunction.filestoanalyse(nameDirectory,'*IMAGE_*')
	print ('nAllFile = ', nAllFile)

	### Introduction variables ###
		### number processor 
	num_cores = multiprocessing.cpu_count()

		### plot latex
	plt.rc('text', usetex=False)

		### Dictionary for all general variable
	dicValueFile = {}
	for nFiles in range(sizeAllFile) :
			# Recherche sous dossier
		subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) ; os.chdir(subDirectoryVideo)

			# lecture data
		(nameVideoData, sizeVideoData) = basefunction.filestoanalyse(subDirectoryVideo, '*video*')
		genData = np.loadtxt('Distance.txt')

			# creating dictionary containing general data (I know it's stupid XD)
		dicValueFile['variableForFile{0}'.format(nFiles)] = basefunction.TraitementFile(nFiles, nAllFile[nFiles], genData, sizeVideoData)




###############################################################################
	### Particle detection ###
	answer = basefunction.question('Do you want to detect the position of the rbcs in data : (1(yes)/0(no) ?')

	if answer is '1' :
		subAnswer = basefunction.question('Do you want to see what the image analysis give ?')
		print('	we will calculate the position of the Rbc in the different video')
		for nFiles in range(sizeAllFile) :
			subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles]) ; os.chdir(subDirectoryVideo)
			(nameVideoData, sizeVideoData) = basefunction.filestoanalyse(subDirectoryVideo, '*video*')
			
			for nFilm in range(sizeVideoData) :
				if nFilm==0 :
					reperageglobule.particledetection(nFilm, nameVideoData, subAnswer)
				else :
					reperageglobule.particledetection(nFilm, nameVideoData, '0')
		print('	the RBC have been detected in all the videos and the data are saved, we\'re going toward tracking')		   

	else :
		print('	ok !!!')




###############################################################################
	### Analysis ###
	answerAnalysis = basefunction.question('Do you want to do an analysis (tracking, velocity, analysis...) (1(yes)/0(no)) ?')


		### Tracking
	if answerAnalysis == '1' :
		answer = basefunction.question('	Do you want to track the RBC (1(yes)/0(no)) ?')

		if answer == '1' :
			print('		we will track the Rbc in the different video')
			for nFiles in range(sizeAllFile) :
				subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
				os.chdir(subDirectoryVideo)
				(nameVideoData, sizeVideoData) = basefunction.filestoanalyse(subDirectoryVideo, '*distanceDataFile*')
				joblib.Parallel(n_jobs=sizeVideoData)( joblib.delayed( globuletracking.particletracking)(nFilm, nameVideoData,
										dicValueFile["variableForFile{0}".format(nFiles)], subDirectoryVideo)
										for nFilm in range(sizeVideoData) )
				print('		the tracking is done !')

		else :
			print('		the tracking is already done !')


		### Local analysis
		answer = basefunction.question('	Do you want to do a local analysis in data 1(yes)/0(no) ?')

		if answer == '1' :
			print('		we will do a local analysis for the different video')
			joblib.Parallel(n_jobs=sizeAllFile)( joblib.delayed(analysis.localanalysis)(nFiles, nameDirectory, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
									  for nFiles in range(sizeAllFile) )
			print('		the local analysis is done !')
		else :
			print('		the local analysis is already done !')


		### Velocity analysis
		answer = basefunction.question('	Do you want to analyse the velocity of the RBC (1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will analyse the velocity of the Rbc in the different video')
			for nFiles in range(sizeAllFile) :
				subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[nFiles])
				os.chdir(subDirectoryVideo)
				(nameVideoData, sizeVideoData) = basefunction.filestoanalyse(subDirectoryVideo, '*trackDataFile*')
				joblib.Parallel(n_jobs=sizeVideoData)( joblib.delayed( velocityanalysis.velocityanalysis)(nFilm, nameVideoData,
										dicValueFile["variableForFile{0}".format(nFiles)], subDirectoryVideo )
										for nFilm in range(sizeVideoData) )
				print('		the analysis is done !')
		else :
			print('		ok !')


		### Plot velocity
		answer = basefunction.question('	Do you want to plot the velocity of the RBC (1(yes)/0(no) ?')
		if answer == '1' :
			joblib.Parallel(n_jobs=sizeAllFile)( joblib.delayed(velocityanalysis.plotvelocity)(nFiles, nameDirectory, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
									  for nFiles in range(sizeAllFile) )
		else :
			print('		ok !')


		### Global analysis
		answer = basefunction.question('	Do you want to do a global analysis in data 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will do a global analysis for the different video')
			joblib.Parallel(n_jobs=sizeAllFile)( joblib.delayed(analysis.globalanalysis)(nameDirectory, nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
									  for nFiles in range(sizeAllFile) )
			print('		the global analysis is done !')


		else :
			print('		the global analysis is done !')

	else :
		print('	ok !!!')




###############################################################################
	### Brute force ###
	answerBruteForce = basefunction.question('Do you want to do a brute force fitting (1(yes)/0(no)) ?')

		### Residual with brute force ###
	if answerBruteForce == '1' :
		answer = basefunction.question('	Do you want to measure the residual with brute force 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will calculate the residual')
			(X,Y) = parameterdispersion.meshgridbrutemethod(nameDirectory)
			joblib.Parallel(n_jobs=sizeAllFile)( joblib.delayed(parameterdispersion.brutemethod)(nameDirectory ,nFiles,
							 			nAllFile, dicValueFile['variableForFile{0}'.format(nFiles)], X, Y)
									  	for nFiles in range(sizeAllFile) )
			print('		residual has been calculated !')
		else :
			print('		residual has been already calculated !')

		### Sum Residual
		answer = basefunction.question('	Do you want to sum all residual 1(yes)/0(no) ?')

		if answer == '1':
			parameterdispersion.sumresidu(nameDirectory, nAllFile, sizeAllFile)
			print('		sum of all residual has been ploted !')
		else:
			print('		ok !')

		### Plot residual
		answer = basefunction.question('	Do you want to plot the residual 1(yes)/0(no) ?')
		if answer == '1' :
			# Search grid
			print('		we will plot the residu')
			(X,Y) = parameterdispersion.searchgrid(nameDirectory)

			# Create commune figure
			fig = plt.figure('plot_all_residu') ; ax = fig.add_subplot(111)
			ax.set_xlabel(r'$\delta$') ; ax.set_ylabel(r'$\xi$')

			# Aplly function
			for nFiles in range(sizeAllFile) :
				ax = parameterdispersion.plotresidu(nameDirectory ,nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)], ax, X, Y)

			# Complete plot with minimum residual
			os.chdir(nameDirectory)
			valueResidual = np.loadtxt('value residual.txt')
			ax.plot(valueResidual[0], valueResidual[1], marker='o', markersize=6, color='red')

			# Save commune figure
			ax.set_yscale('log')
			fig.savefig('residual.pdf', frameon=False, bbox_inch='tight', orientation='landscape')
			plt.show()
			plt.close(fig)
		
			print('		residual has been plotted !')
		else :
			print('		ok !')

	else :
		print('	ok !!!')






###############################################################################
	### y fit ###
	answerFitGlobal = basefunction.question('Do you want to fit the global data (1(yes)/0(no)) ?')

		### y(x) fit
	if answerFitGlobal == '1' :
		answer = basefunction.question('	Do you want to fit y(x) 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will fit y(x)')
			for nFiles in range(sizeAllFile) :
				fitglobal.fityxglobal(nameDirectory ,nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else :
			print('		ok !')

		### a(x) fit
		answer = basefunction.question('	Do you want to fit y(x/w)/w 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will fit y(x/w)/w')
			for nFiles in range(sizeAllFile) :
				fitglobal.fitaxglobal(nameDirectory ,nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else :
			print('		ok !')

		### y(x) fit by Olla
		answer = basefunction.question('	Do you want to fit y(x) by Olla 1(yes)/0(no) ?')
		if answer == '1':
			print('		we will fit y(x)')
			for nFiles in range(sizeAllFile):
				fitglobal.fityollaglobal(nameDirectory, nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else:
			print('		ok !')

		### y(tN) fit
		answer = basefunction.question('	Do you want to fit y(tN) 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will fit y(tN)')
			for nFiles in range(sizeAllFile) :
				fitglobal.fitytNqin(nameDirectory ,nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else :
			print('		ok !')

		### log(y(log(tN))) fit
		answer = basefunction.question('	Do you want to fit log(y(log(tN))) 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will fit y(tN)')
			for nFiles in range(sizeAllFile) :
				fitglobal.fitlogytN(nameDirectory ,nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else :
			print('		ok !')

		### fit test
		answer = basefunction.question('	Do you want to fit all the data test 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will fit test')
			for nFiles in range(sizeAllFile):
				fitglobal.fittestglobal(nameDirectory ,nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else :
			print('		ok !')

		### fit y(x)
		answer = basefunction.question('	Do you want to fit y(x), y0... 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will fit test')
			for nFiles in range(sizeAllFile):
				fitglobal.fityx(nameDirectory ,nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else :
			print('		ok !')

		### fit y(tN) direct
		answer = basefunction.question('	Do you want to fit y(tN) by a direct method 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will fit test')
			for nFiles in range(sizeAllFile):
				fitglobal.fitydirect(nameDirectory ,nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else :
			print('		ok !')

	else :
		print('	ok !!!')



	### Total fit ###
	answerFitTotal = basefunction.question('Do you want to fit all the data together (1(yes)/0(no)) ?')
		### a(b) fit
	if answerFitTotal == '1' :
		answer = basefunction.question('	Do you want to fit all the data y(x) 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will fit y(x)')
			fittotal.fitaxtotal(nameDirectory, nAllFile, sizeAllFile, dicValueFile)
			print('		the fit is done !')
		else :
			print('		ok !')

		### y(tN) fit
		answer = basefunction.question('	Do you want to fit all the data y(tN) 1(yes)/0(no) ?')
		if answer == '1' :
			print('		we will fit y(tN)')
			fittotal.fitytNtotal(nameDirectory, nAllFile, sizeAllFile, dicValueFile)
			print('		the fit is done !')
		else :
			print('		ok !')

	else :
		print('	ok !!!')



	### fit points ###
	answerFitPoints = basefunction.question('Do you want to fit the points (1(yes)/0(no)) ?')
		### y point
	if answerFitPoints == '1' :
		answer = basefunction.question('	Do you want to fit the point  y(x) (1(yes)/0(no)) ?')
		if answer == '1' :
			print('		we will fit y(x)')
			for nFiles in range(sizeAllFile) :
				fitpoint.fitypoint(nameDirectory, nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else :
			print('		ok !')

		### a points
		answer = basefunction.question('	Do you want to fit the point  y(x/w)/w (1(yes)/0(no)) ?')
		if answer == '1' :
			print('		we will fit y(x)')
			for nFiles in range(sizeAllFile) :
				fitpoint.fitapoint(nameDirectory, nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)])
			print('		the fit is done !')
		else :
			print('		ok !')

	else :
		print('	ok !!!')



	### Fit by monte carlo method ###
	answerMonteCarlo = basefunction.question('Do you want to use a Monte Carlo method (1(yes)/0(no)) ?')
		### y mean
	if answerMonteCarlo == '1' :
		answer = basefunction.question('	Do you want to use a monte carlo method to determine delta xi for y(x) (1(yes)/0(no)) ?')
		if answer == '1' :
			print('		we will use a monte carlo method')
			joblib.Parallel(n_jobs=sizeAllFile)( joblib.delayed(parameterdispersion.montecarlofity)
												 (nameDirectory, nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)]) for nFiles in range(sizeAllFile) )
			print('		the method has been applied !')
		else :
			print('		ok !')

		### all point
		answer = basefunction.question('	Do you want to use a monte carlo method to determine lambda and xi for y(tN) (1(yes)/0(no)) ?')
		if answer == '1' :
			print('		we will use a monte carlo method for the point y(tN)')
			joblib.Parallel(n_jobs=sizeAllFile)( joblib.delayed(parameterdispersion.montecarlofitytN)
												 (nameDirectory, nFiles, nAllFile, dicValueFile["variableForFile{0}".format(nFiles)]) for nFiles in range(sizeAllFile) )
			print('		the method has been applied !')
		else :
			print('		ok !')

	else :
		print('	ok!!!')

###############################################################################
	plt.show()


