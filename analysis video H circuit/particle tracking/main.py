import multiprocessing ;
import joblib ;						

import os ;

import numpy as np ; 

import matplotlib.pyplot as plt ;

import fonctionbase ;
import particledetection ;
import particletracking ;
import velocityanalysis ;
import searchpulseform ;
import analysetau ;





############################################
######### Classe basic valeur data #########
############################################

class TraitementFile :

	def __init__(self, nFiles, nameFile, base_velocity, distance) :
		self.color_file_video = np.array(['midnightblue', 'navy', 'darkblue', 'blue', 'mediumblue', 'royalblue', 
							'dodgerblue', 'cornflowerblue', 'skyblue', 'cyan', 'deepskyblue', 'lightskyblue', 
							'aqua', 'seagreen', 'turquoise', 'paleturquoise', 'aquamarine',
                             'darkgreen', 'forestgreen', 'green', 'limegreen', 'lime', 'darkseagreen',
							 'chartreuse', 'greenyellow', 'lawngreen', 'yellow', 'gold', 'orange', 'goldenrod',
                             'darkred', 'maroon', 'red', 'firebrick', 'brown', 'coral', 'sandybrown', 'burlywood',
                             'indigo', 'purple','darkviolet', 'violet', 'deeppink','magenta',
                             'black', 'dimgrey', 'darkgrey', 'lightgrey', 'whitesmoke']) ;
		self.color_file = np.array(['blue', 'red', 'orange', 'yellow', 'violet', 'turquoise']) ;
		self.mark_file = np.array(['o', 'x', '^', 's']) ; 
		self.line_style_file = np.array(['-', '--', '-.', ':']) ;

		self.n_files = nFiles
		self.name_file = nameFile

		self.liaison_pix_to_um = 5.86e-6/10 ;
		self.reduction_picture = 100 ;
		self.limit_Intensity = 170 ;
		self.search_zone_Y = 1.25 ;
		
		self.limit_pulse = np.array([0.1, 0.5, 0.9]) ;
		self.limit_velocity = np.array([90, 50, 10]) ;
		#self.limit_pulse = np.array([0.8, 0.82, 0.84, 0.86, 0.88, 0.9]) ;
		#self.limit_velocity = np.array([20, 18, 16, 14, 12, 10]) ;
		self.base_velocity = base_velocity ;		
		
		self.interspace_pulse = 10 ;
		self.max_choix = 100 ;
		self.histogramm_bin = 400 ;
		self.histogramm_range = 400 ;

		self.rayon_globule_rouge = 2.8e-6 ;
		self.distance = distance ;
		self.viscosity = np.array([1.7, 5.0, 9.0]) ;
		




################################
######### Main program #########
################################

if __name__ == '__main__':
###################################################################################################
	### Change directory ###
	nameDirectory = 'E:/diff_height_2/'
	while os.path.isdir(nameDirectory) == False :
		print('   Error : your directory doesn t exist, please indicate a correct path')
		nameDirectory = input('Indicate the place of your data (E:/.../) : ') ;
	print('   your directory exist !')
	os.chdir(nameDirectory) ;


	### Search Sub directory ###
	(nameAllFile, sizeAllFileToCompare) = fonctionbase.filestoanalyse(nameDirectory,'*IMAGE*') ;
	print ('nameAllFile = ', nameAllFile) ;


	### Introduction variables ###
	dictionnaryForVariable = {}

	for nFiles in range(sizeAllFileToCompare) :
		subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
		os.chdir(subDirectoryVideo) ;

		generalDataDistance = np.loadtxt('Distance.txt') ;
		baseVelocity = np.loadtxt('baseVelocity.txt') ;

		dictionnaryForVariable["variableForFile{0}".format(nFiles)] = TraitementFile(nFiles, nameAllFile[nFiles], baseVelocity, generalDataDistance) ;






###################################################################################################
	### Control of the utility of the particle detection in velocity file ###
	particleTrackingAnswer = int(input('Do you want to detect the position of the rbcs in velocity data : (1(yes)/0(no))')) ;
	while particleTrackingAnswer != 1 and particleTrackingAnswer != 0 :
		particleTrackingAnswer = int(input('   Error : please write 1 or 0'))

	### particle detection ###
	if particleTrackingAnswer == 1 :
		print('    we will calculate the position of the Rbc in the different video')
	
		for nFiles in range(sizeAllFileToCompare) :
			subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
			os.chdir(subDirectoryVideo) ;

			(nameVideoData, sizeVideoData) = fonctionbase.filestoanalyse(subDirectoryVideo, '*video*') ;
			print(nameVideoData)

			num_cores = multiprocessing.cpu_count() ; print('the number of cores wich could be used is : ', num_cores)
			joblib.Parallel(n_jobs=4)( joblib.delayed(particledetection.particledetection)(nFilm, nameVideoData, dictionnaryForVariable["variableForFile{0}".format(nFiles)]) for nFilm in range(sizeVideoData) )
		print('   the detection is done !')
		print('   we will continue the operation (going toward tracking)')

	else :
		print('   we will continue the operation (going toward tracking)')





###################################################################################################
	### Control of the utility of particle tracking ###
	particleTrackingAnswer = int(input('Do you want to track the rbcs : (1(yes)/0(no))')) ;
	while particleTrackingAnswer != 1 and particleTrackingAnswer != 0 :
		particleTrackingAnswer = int(input('   Error : please write 1 or 0'))

	### Track of the particle ###
	if particleTrackingAnswer == 1 :
		print('   we will track the Rbc in the different video')
	
		for nFiles in range(sizeAllFileToCompare) :
			subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
			os.chdir(subDirectoryVideo) ;

			(nameDataDetection, sizeDataDetection) = fonctionbase.filestoanalyse(subDirectoryVideo, '*dataFrameDetectionVelocityFile*') ;

			num_cores = multiprocessing.cpu_count() ; print('the number of cores used is : ', num_cores)
			joblib.Parallel(n_jobs=12)( joblib.delayed(particletracking.particletracking)(nameDataDetection, nDataFile, dictionnaryForVariable["variableForFile{0}".format(nFiles)]) 
							  for nDataFile in range(sizeDataDetection) )

		print('   the tracking is done !')
		print('   we will continue the operation (going toward velocity)')

	else :
		print('   we will continue the operation (going toward velocity)')
	





###################################################################################################
	### Control of the utility of the velocity analysis ###
	velocityAnalysisAnswer = int(input('Do you want to calculate the velocity of the rbcs : (1(yes)/0(no))')) ;
	while velocityAnalysisAnswer != 1 and velocityAnalysisAnswer != 0 :
		velocityAnalysisAnswer = int(input('   Error : please write 1 or 0'))

	if velocityAnalysisAnswer == 1 :
		print('   we will calculate the velocity of the Rbc in the different video')

		for nFiles in range(sizeAllFileToCompare) :
		# Opening of the file
			subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
			os.chdir(subDirectoryVideo) ;

		# Search data
			(nameDataTrack, sizeDataTrack) = fonctionbase.filestoanalyse(subDirectoryVideo, '*dataFrameTrackFile*') ;

		# Velocity analysis
			num_cores = multiprocessing.cpu_count() ; print('the number of cores used is : ', num_cores)
			joblib.Parallel(n_jobs=num_cores)( joblib.delayed(velocityanalysis.velocityanalysis)(nameDataTrack, nDataFile, dictionnaryForVariable["variableForFile{0}".format(nFiles)]) 
								for nDataFile in range(sizeDataTrack) )

		# Plot
			print('	ploting velocity')
			velocityanalysis.plotvelocity(subDirectoryVideo, dictionnaryForVariable["variableForFile{0}".format(nFiles)])

			print('   the velocity has been calculated and plotted')
			print('   we will continue the operation (going toward counting the number of Rbc per picture in video)')


	else :
		for nFiles in range(sizeAllFileToCompare) :
		# Opening of the file
			subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
			os.chdir(subDirectoryVideo) ;

			print('	ploting velocity')
			#velocityanalysis.plotvelocity(subDirectoryVideo, dictionnaryForVariable["variableForFile{0}".format(nFiles)])
			print('   we will continue the operation (going toward counting the number of Rbc per picture in video)')






############################################################################################################################
	### Control of the utility of the measure of the lenght of the pulse ###
	measuringLenghtPulseAnswer = int(input('Do you want to have the form of the pulse : (1(yes)/0(no))')) ;
	while measuringLenghtPulseAnswer != 1 and measuringLenghtPulseAnswer != 0 :
		measuringLenghtPulseAnswer = int(input('		Error : please write 1 or 0'))

	if measuringLenghtPulseAnswer == 1 :
		print('   we will measure the general lenght')

		for nFiles in range(sizeAllFileToCompare) :
		# Opening of the file
			subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
			os.chdir(subDirectoryVideo) ;

		# Search data
			(nameDataVelocity, sizeDataVelocity) = fonctionbase.filestoanalyse(subDirectoryVideo, '*velocityLocalFile*') ;

		# Velocity analysis
			num_cores = multiprocessing.cpu_count() ; print('the total number of cores is : ', num_cores)
			joblib.Parallel(n_jobs=num_cores)( joblib.delayed(searchpulseform.lenghtpulseforme)( nameDataVelocity, nDataFile, dictionnaryForVariable["variableForFile{0}".format(nFiles)]) 
									 for nDataFile in range(sizeDataVelocity) )

		# Plot
			searchpulseform.plotformepulse(subDirectoryVideo, dictionnaryForVariable["variableForFile{0}".format(nFiles)]) 

	else :
		for nFiles in range(sizeAllFileToCompare) :
		# Opening of the file
			subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
			os.chdir(subDirectoryVideo) ;
			#searchpulseform.plotformepulse(subDirectoryVideo, dictionnaryForVariable["variableForFile{0}".format(nFiles)]) 


	print('   we will continue the operation (going toward the measure of tau)')






###################################################################################################
	### Control of the utility of the measure of tau ###
	measuringTauAnswer = int(input('Do you want to measure tau in s and the velocity in um/s : (1(yes)/0(no))')) ;
	while measuringTauAnswer != 1 and measuringTauAnswer != 0 :
		measuringTauAnswer = int(input('		Error : please write 1 or 0'))

	if measuringTauAnswer == 1 :
		print('   we will measure tau')

		for nFiles in range(sizeAllFileToCompare) :
		# Opening of the file
			subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
			print(subDirectoryVideo)
			os.chdir(subDirectoryVideo) ;

		# Search data
			velocity0 = np.loadtxt('maximalVelocity.txt') ; 
			lenghtPulse = np.loadtxt('globalLenghtData.txt') ;
			velocity = np.loadtxt('velocityForY.txt')

			analysetau.datareorganisation(velocity, lenghtPulse, velocity0, dictionnaryForVariable["variableForFile{0}".format(nFiles)]) 

	
	print('   we will continue the operation (going toward measure of the fit of tau)')










###################################################################################################
	### Control of the fit of tau ###
		# Create new data
	dataForGlobalFit = {} ;

	for nFiles in range(sizeAllFileToCompare) : #sizeAllFileToCompare) :
		# Opening of the file
		subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
		os.chdir(subDirectoryVideo) ;

		# Search data
		Vglobal = np.loadtxt('Vglobal.txt') ; 
		tauExp = np.loadtxt('tauExp.txt') ;
		V0global = np.loadtxt('V0global.txt') ;
		X1global = np.loadtxt('X1global.txt') ;

		dataForGlobalFit = analysetau.measuretau(Vglobal, tauExp, V0global, X1global, dictionnaryForVariable["variableForFile{0}".format(nFiles)], dataForGlobalFit) ;
		dataForGlobalFit = analysetau.measureX(Vglobal, tauExp, V0global, X1global, dictionnaryForVariable["variableForFile{0}".format(nFiles)], dataForGlobalFit) ;


	print('   we will continue toward fit, necessity to fit all the data together ?')







###################################################################################################
	### Control of the utility of the measure of tau ###
	fitTauTotal = int(input('Do you want to all the data together : (1(yes)/0(no))')) ;
	while fitTauTotal != 1 and fitTauTotal != 0 :
		fitTauTotal = int(input('		Error : please write 1 or 0'))

	if fitTauTotal == 1 :
		print('   we will fit all the data together')

		# Opening of the file
		subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles]) ;
		print(subDirectoryVideo)
		os.chdir(subDirectoryVideo) ;

		analysetau.fitalldata(dataForGlobalFit, dictionnaryForVariable) 
	
	print('//      it s finished !?!')

	
		
		
		
	plt.show()
