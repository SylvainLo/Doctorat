import os ;
import glob ;

import numpy as np ;


###############################
######### Search File #########
###############################

	######### Search File #########

def filestoanalyse(directory, nameToSearch) :
	os.chdir(directory) ;                 # Change the current working directory to path
	listFiles = glob.glob(nameToSearch) ;  # New vector containing string of the videos in the file
	nbrFiles = np.size(listFiles) ;        # The number of video

	return (listFiles, nbrFiles)



#######################################
######### For multiprocessing #########
#######################################

		######### For multiprocessing #########

def compute(n):
    import time, socket
    time.sleep(n)
    host = socket.gethostname()
    return (host, n)