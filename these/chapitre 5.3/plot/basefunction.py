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



#######################################################
######### Aproximation value by Taylor series #########
#######################################################
def extrapolation(f, g) :
	"""f(g) = f(a) + f\'(a)(x-a) + f\'\'(a)/2(x-a)^2
		f(0) = f(x1) - x1*(-3f(x1)+4f(x2)-f(x3))/(2(x2-x1)) + x1^2*(2f(x1)-5f(x2)+4(f(x3)-f(x4))/(2(x2-x1)^2)
	"""

	k = f[1] - g[1] * (-3*f[1]+4*f[2]-f[3])/(2*(g[2]-g[1])) + g[1]**2 * (2*f[1]-5*f[2]+4*f[3]-1*f[4])/(2*(g[2]-g[1])**2)


	return k

###############################
######### Search File #########
###############################
def exvelocity(x) :
	print('measure the velocity of the {} of the pulse : \n'
	' - choose a picture from the {} of the pulse (there is a need to have multiple RBC\n'
	' - select a rectangle containing some RBC the result of the correlation with the next picture should appear\n'
	' - if you\'re ok with the result press esc, if not draw a new rectangle'.format(x, x))
