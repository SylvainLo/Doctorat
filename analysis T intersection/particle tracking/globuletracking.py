import numpy as np

import os



#######################################################
######### Particle tracking in velocity data #########
#######################################################
def particletracking(nFilm, nameVideoData, position, subDirectoryVideo) :
	os.chdir(subDirectoryVideo)
	data = np.loadtxt(nameVideoData[nFilm]) ; print(nameVideoData[nFilm]) 

	### introduction constant ###
		# constant
	searchXP = position.distance[nFilm,5] ; searchX = searchXP
	searchYP = position.distance[nFilm,6]
	velocityP = position.distance[nFilm,7]

		# creation vector
	(row, column) = np.shape(data)
	datTrack = np.empty((row, column+1))

	numTrack = int(0) ; rowTrack = int(0)
	vectorTest = np.ones(row, dtype=np.bool)
	

	### Tracking
		# search of the first particle being not part of a track
	while rowTrack != row :
		a = np.argmax(vectorTest)

		datTrack[rowTrack,0] = numTrack ; datTrack[rowTrack,1:column+1] = data[a,:]

		x = data[a,2] ; xEx = x + velocityP ; y = data[a,3]
		numFrame = data[a,1] ; rowTrack += 1
		vectorTest[a] = 0
		k = True

		# search of the particle being in this track
		while k == True and rowTrack != row :
			datTest = data[ np.all([data[:,1] == numFrame+1, vectorTest == 1,
									data[:,2] < xEx + searchX, data[:,2] > xEx - searchX,
									data[:,3] < y+searchYP, data[:,3] > y-searchYP], axis=0) ]
			(rowTest,colTest) = np.shape(datTest)

			if rowTest == 1 :
				a = int(datTest[0,0])
				datTrack[rowTrack,0] = numTrack ; datTrack[rowTrack,1:column+1] = datTest

				velocity = datTrack[rowTrack,3] - datTrack[rowTrack-1,3] ; searchX = searchXP / 4
					
				x = data[a,2] ; xEx = x + velocity ; y = data[a,3]
				numFrame = data[a,1] ; rowTrack += 1
				vectorTest[a] = 0
					
			elif rowTest > 1 :
				distanceYbYa = np.abs(datTest[:, 3] - y)
				goodTrack = np.argmin(distanceYbYa)
					
				a = int(datTest[goodTrack, 0])
				datTrack[rowTrack, 0] = numTrack ; datTrack[rowTrack, 1:column+1] = datTest[goodTrack, :]

				velocity = datTrack[rowTrack, 3] - datTrack[rowTrack - 1, 3] ; searchX = searchXP / 4
					
				x = data[a, 2] ; xEx = x + velocity ; y = data[a, 3]
				numFrame = data[a, 1] ; rowTrack += 1
				vectorTest[a] = 0

			else :
				k = False ; numTrack += 1
				searchX = searchXP



	trackFile = 'trackDataFile_%03d.txt'%(nFilm+1)
	np.savetxt(trackFile, datTrack, fmt = '%.0d %.0d %.0d %.8e %.8e %.4f %.4f %.4f %.4f %.4f %4f %4f' , delimiter='	',
			   header="num Track, nParticle, nFrame, pos x , pos y , small d, big d, d Wall//Wall, D Wall//Wall bot, big radius, small radius, angle")

	return

