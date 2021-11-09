import numpy as np  # print('numpy imported')
import os  # print('os imported')
import glob  # print('glob imported')


###############################
######### Search File #########
###############################
def filestoanalyse( directory, nameToSearch ):
	os.chdir(directory)  # Change the current working directory
	listFile = glob.glob(nameToSearch)  # New vector containing string of the nameToSearch in the file
	nbrFile = np.size(listFile)  # The number of video

	return listFile, nbrFile


###############################
######### Search File #########
###############################
def question( sentence ):
	while True:
		partAnswer = input(sentence)

		if partAnswer == '1' or partAnswer == '0':
			break

	return partAnswer


##########################################
######### Class basic value data #########
##########################################
class TraitementFile:

	def __init__( self, nFiles, nameFile, basicProperty, numberMovie ):
		# part graph same for all graphic
		self.color_file = np.array(
			['navy', 'dodgerblue', 'turquoise', 'paleturquoise', 'darkgreen', 'chartreuse', 'y', 'orange',
			 'darkred', 'red', 'purple', 'peru', 'deeppink', 'black', 'dimgray'])
		self.mark_file = np.array(['o', 'x', '^', 's'])
		self.line_style_file = np.array(['-', '--', '-.', ':'])

		# name data
		self.n_files = nFiles
		self.name_file = nameFile
		self.number_movies = numberMovie

		# basic data
		self.distance = basicProperty

		# separation general data
		self.delta_x = 44e-6  # 8 point per movie


####################################################################
######### function to create a vector containing all data  #########
####################################################################

def dicalldata( nameDirectory, nAllFile, sizeAllFile ):
	### Creation variable with all global data
	dicAllData = {}

	for i in range(sizeAllFile):
		# Change of directory
		subDirectoryVideo = '{0}/{1}'.format(nameDirectory, nAllFile[i])
		os.chdir(subDirectoryVideo)

		# Lecture data
		datGlobal = np.loadtxt('globalAnalysisData.txt')
		vMax = np.loadtxt('V-R.txt')

		# remplissage dictionnaire
		dicAllData['data_{}'.format(i)] = datGlobal
		dicAllData['V0_{}'.format(i)] = vMax

	os.chdir(nameDirectory)

	return dicAllData
