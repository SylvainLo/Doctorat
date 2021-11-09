import numpy as np

import os

import matplotlib.pyplot as plt
import matplotlib2tikz as tikz_save
import matplotlib as mpl

import multiprocessing as mp

import basefunction
import figurefitlibre
import plotvelocity
import ximin
import plotresult
import figuredeltaxi



###############################
####### Variable global #######
###############################
	### variable
N = 5
R0 = 2.8e-6
num_cores = mp.cpu_count() - 1
col_velocity = np.array(['gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray'])

	### parameter plot
font = {'size':10}
plt.rc('font', **font)
rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(rc_params)





##############################
######### Main code ##########
##############################

if __name__ == '__main__':
###################################################################################
### Change directory ###
	nameDirectory = 'F:/Migration_GR_seul'
	while os.path.isdir(nameDirectory) is False:
		print('   Error : your directory doesn\'t exist, please indicate a correct path')
		nameDirectory = input('Indicate the place of your data (E:/.../) : ')

	os.chdir(nameDirectory)
	print('   your directory exist !')



### Search Sub directory ###
	(nameAllFile, sizeAllFile) = basefunction.filestoanalyse(nameDirectory, '*IMAGE_*')
	print('nAllFile = ', nameAllFile)



### Introduction variables ###
	### Dictionary for all general variable
	dicData = {}
	for nFiles in range(sizeAllFile):
		# Recherche sous dossier
		subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nameAllFile[nFiles])
		os.chdir(subDirectoryVideo)

		# lecture data
		(nameVideoData, sizeVideoData) = basefunction.filestoanalyse(subDirectoryVideo, '*video*')
		genData = np.loadtxt('Distance.txt')

		# creating dictionary containing general data (I know it's stupid XD)
		dicData['{}'.format(nFiles)] = {}
		dicData['{}'.format(nFiles)]['baseData'] = basefunction.TraitementFile(nFiles, nameAllFile[nFiles],
																						genData, sizeVideoData)
		dicData['{}'.format(nFiles)]['data'] = np.loadtxt('globalAnalysisData.txt')
		dicData['{}'.format(nFiles)]['velocity'] = np.loadtxt('velDataTotal.txt')
		dicData['{}'.format(nFiles)]['residual'] = np.load('matriceResidual.npy')
		dicData['{}'.format(nFiles)]['color_velocity'] = col_velocity[nFiles]

	os.chdir(nameDirectory)





###############################################################################
### Velocity ###
	### Question
	answerVelocity = basefunction.question('Do you want to plot the velocity of the RBC (1(yes)/0(no) ?')


	### Plot velocity
	if answerVelocity == '1':
		print('	we will plot the velocity')

		figV = plt.figure('plotvelocity', figsize=[5.3,7])
		figV.subplots_adjust(0.12, 0.08, 0.97, 1, 0.15, 0.05)

		for i in range(4) :
			figV = plotvelocity.plotvelocity(dicData['{}'.format(i)], figV, i+1, True)

		#figV = plotvelocity.plotvelocity(dicData['3'], figV, 4, True)
		#figV = plotvelocity.plotvelocity(dicData['4'], figV, 4, False)

		for i in range(7,9) :
			figV = plotvelocity.plotvelocity(dicData['{}'.format(i)], figV, i-2, True)

		print('		the velocity has been plotted')


		# save first plot
		figV.savefig('velocity_all_data_papier.svg', format='svg')
		figV.savefig('velocity_all_data_papier.pdf', format='pdf')

	else:
		print('		ok !')







###############################################################################
### Plot multiple fit on 20um ###
	### Question
	answerMultipleFit = basefunction.question('Do you want to plot multiple fit on data with residu (1(yes)/0(no)) ?')


	### Plot analysis
	if answerMultipleFit == '1':
		print('		we will apply multiple fit on data 20um')
		figlibre = figurefitlibre.figPlot(dicData['2'], nameDirectory)
		print('		the local analysis is done !')

		# save first plot
		figlibre.savefig('plot_multiple_fit.svg', format='svg')
		figlibre.savefig('plot_multiple_fit.pdf', format='pdf')

	else:
		print('		ok !')







###############################################################################
### Search minimum xi ###
	### Question
	answerXiMin = basefunction.question('Do you want to plot xi in function delta (1(yes)/0(no)) ?')

	### Search xi minimal
	if answerXiMin == '1' :

		# plot
		figXi = ximin.plotxi()

		figXi.savefig( 'delta_xi.svg', format='svg' )
		figXi.savefig( 'delta_xi.pdf', format='pdf' )

	else :
		print('ok')







###############################################################################
### Plot xi and delta to show no correlation ###
	### Question
	answerCorrelation = basefunction.question('Do you want to plot xi and delta no correlation (1(yes)/0(no)) ?')

	### Search xi minimal
	if answerCorrelation == '1' :

	### Plot
		figCor = figuredeltaxi.plotCorrelation(dicData)

		figCor.savefig( 'correlation_delta_xi.svg', format='svg' )
		figCor.savefig( 'correlation_delta_xi.pdf', format='pdf' )

	else :
		print('ok')





###############################################################################
### Fit ###
	### Question
	answerFit = basefunction.question('Do you want to fit the data (1(yes)/0(no)) ?')

	if answerFit == '1':
		print('		we will fit y(x)')

	### y(x) fit
		# plot
		figD = plt.figure('plotresult', figsize=[5.3,7])
		figD.subplots_adjust(0.12, 0.07, 0.95, 0.95, 0.35, 0.28)

		# value delta and xi
		valDelXi = np.loadtxt('valMin_delta-xi.txt')
		xiRel = np.loadtxt('gridXiMean.txt')
		gridDelta = np.loadtxt('gridDelta.txt')

		for i in range(4):
			figD = plotresult.fitaxtotal(dicData['{}'.format(i)], figD, valDelXi, i+1)

		for i in range(7, 9):
			figD = plotresult.fitaxtotal(dicData['{}'.format(i)], figD, valDelXi, i-2)


		# save first plot
		figD.savefig('fit_all_data_papier.svg', format='svg')
		figD.savefig('fit_all_data_papier.pdf', format='pdf')

		print('		the fit is done !')

	else:
		print('		ok !')





###############################################################################
### Fit ###
	### Question
	answerFitLH = basefunction.question('Do you want to fit the data for RBC light and heavy (1(yes)/0(no)) ?')


	if answerFitLH == '1':
		print('		we will fit y(x)')

	### y(x) fit
		# plot
		figHL = plt.figure('plotresultheavylight', figsize=[5,4])
		figHL.subplots_adjust( 0.07, 0.07, 0.95, 0.95, 0.4, 0.4 )

		# value delta and xi
		valDelXi = np.loadtxt('valMin_delta-xi.txt')

		# plot fit
		figHL = plotresult.fitaxtotalhl(dicData['4'], dicData['5'], dicData['6'], figHL, valDelXi)


		# save first plot
		tikz_save.save('heavy_light_GR_papier.tex', figure=figHL, figureheight='\\figureheighthl',
						   figurewidth='\\figurewidthhl')
		figHL.savefig('heavy_light_GR_paper.svg', format='svg')
		figHL.savefig('heavy_light_GR_paper.pdf', format='pgf')

		print('		the fit is done !')

	else:
		print('		ok !')





###############################################################################
 ### Show plot ###
	plt.show()


