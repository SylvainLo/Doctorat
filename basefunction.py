import numpy as np 				#print('numpy imported')
import os 					#print('os imported')
import glob



###############################
######### Search File #########
###############################
def filestoanalyse(directory, nameToSearch) :
	os.chdir(directory)                  # Change the current working directory
	listFile = glob.glob(nameToSearch)   # New vector containing string of the nameToSearch in the file
	nbrFile = np.size(listFile)          # The number of files

	print('	In your folder', directory, 'the files are : ', listFile)
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

#########################################
######### control presence file #########
#########################################
def control(dir) :
	while os.path.isdir( dir ) is False :
		print( '   Error : your directory does not exist !' )
		dir = input( 'Directory of your data (E:/.../) : ' )

	return dir


#########################################
######### control presence file #########
#########################################
def introduction() :
	print( '\n This program allow the analysis of the evolution of a profile of hematocrit after an intersection in T. '
		   '\n \n '
		   'The channel need to be horizontal !!!'
		   '\n \n'
		   'You need to follow a number of requirement to make it function : \n'
		   '	- Your data must be stock in your drive following a certain order and be named following a certain nomenclature \n '
		   '		|-> the directory (the one called in the code) \n'
		   '			|-> inside this directory, we need to find one or different folders containing the word '
		   '\"...IMAGE...\", each folder contain data for different channel for exemple or represent different experimental day \n'
		   '				|-> inside the folders \"...IMAGE...\", we need to find one or different folder containing \"...%...\", '
		   'each folder contain the video for different hematocrit for exemple \n'
		   '					|-> in this folder \"...%...\", the different video (in format avi) and a file named '
		   '\"experimental_value.txt\" are present'
		   '\n'
		   '	- The file \"experimental_value.txt\" must follow this rule : \n'
		   '		|-> the first line contain the title of the column : numero of the video, x position (um), heignt (um), '
		   'velocity (pix/frame), magnification, size pixel on camera(um)\n'
		   '		|-> the other lines contain the data for each video'
		   '\n \n'
		   'At the beginning, you will be asked if you want to calculate different experimental values. If the file containing'
		   'this values does not exist, the program will measure them by default an create a file containing the result. '
		   'Otherwise it will take into consideration the result of the question.'
		   '\n \n'
		   'If there is some problem contact losserand.sylvain@gmail.com with a copy of the error message.\n')

	return



#########################################
######### control presence file #########
#########################################
def backgroundrule() :
	print( '\n A background of the video is extracted, this background need some manual correction : in the picture '
		   '\"threshold background\" you need to select an area outside the channel (the area shouldn\'t have a part of the wall).'
		   'An other picture of the area should appear, if the result is correct, press escape or choose an othe area in the'
		   'initial picture \n \n'
		   )

	return




#########################################
######### control presence file #########
#########################################
def wallrule() :
	print( '\n From the background, we extract the position of the superior wall and the inferior, for this you need to move the '
		   'bar threshold on the picture \"threshold\" and look at the result on the picture \"wall\" the result (the wall are '
		   'the lines in black). Don\'t go at too high threshold or you risk to plant the program. \n \n'
		   )

	return


#########################################
######### control presence file #########
#########################################
def linerule() :
	print( '\n Using the ruler you can choose the position where the profile will be measured and confirm the position by'
		   'pushing the button \'enter\' (in reality we use an area large of 41 pixel centered on the value y chosen of the line).'
		   'Only one line is chosen per video \n \n'
		   )

	return



#########################################
######### control presence file #########
#########################################
def cflrule() :
	print( '\n Choose the value of the kernel and the value of the threshold so that only red blood cells are seen on the picture'
		   '\"threshold\". Be careful, it\'s extremely important step which can expand or reduce the size of the cfl. But in all '
		   'case the wall and the noise in the background should disappear every time. \n \n'
		   )

	return