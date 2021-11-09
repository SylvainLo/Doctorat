import numpy as np 				#print('numpy imported')
import os 					#print('os imported')
import glob



###############################
######### Search File #########
###############################
def filestoanalyse(directory, nameToSearch) :
	os.chdir(directory)                  # Change the current working directory
	listFile = glob.glob(nameToSearch)   # New vector containing string of the nameToSearch in the file
	nbrFile = np.size(listFile)          # The number of video

	return listFile, nbrFile



###############################
######### Search File #########
###############################
def question(sentence) :
	while True :
		answer = input(sentence)

		if answer == '1' or answer == '0' :
			print('	ok')
			break

	return answer