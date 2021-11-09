import numpy as np ;
import pandas ;




#######################################################
######### Particle tracking in velocity data #########
#######################################################

		######### Particle Tracking #########

def particletracking(nameDataDetection, nDataFile, variableFile) :
	# Loading data
		# Loading global data
	baseVelocity = variableFile.base_velocity 

		# Loading local data	
	dataFrameDetection = pandas.read_csv(nameDataDetection[nDataFile]) ; 
	print(nameDataDetection[nDataFile])

	# Introduction constante
		# variable for boucle
	(rowDetection, column) = np.shape(dataFrameDetection) ;
	numeroTrack = int(0) ; rowTracking = int(0) ;

		# Variable for tracking search
	searchZoneY = variableFile.search_zone_Y ; 
	searchZoneX = baseVelocity[nDataFile,1] ; preciseSearchZoneX = searchZoneX/5 ;
	velocity = baseVelocity[nDataFile,0] ;

		# Creation new DataFrame for saving track
	columnTracking = ['numeroTrack', 'x', 'y', 'frame', 'nParticle']; 
	dataForTracking = pandas.DataFrame(columns=columnTracking) ; 

	# Tracking
	while rowTracking != rowDetection :
		#print(rowTracking)
		# Search first globule of the tracking
		dataForTracking.loc[rowTracking] = [ numeroTrack, dataFrameDetection.x.iloc[0], dataFrameDetection.y.iloc[0], 
										dataFrameDetection.frame.iloc[0], dataFrameDetection.nParticle.iloc[0] ] ; 

		x = dataFrameDetection.x.iat[0] ; xExpected = x + velocity ; y = dataFrameDetection.y.iat[0] ;
		numeroFrame = dataFrameDetection.frame.iat[0] ;

		dataFrameDetection.drop(dataFrameDetection.nParticle.iloc[0], inplace=True) ; 
		rowTracking += 1 ; k = True ; searchZoneXReal = searchZoneX ; 
		
		# Search candidate for the tracking
		while k == True and rowTracking != rowDetection :
			dataFrameTest = dataFrameDetection[ (dataFrameDetection.frame == numeroFrame+1) & 
								(dataFrameDetection.x < xExpected + searchZoneXReal) & 
								(dataFrameDetection.x > xExpected - searchZoneXReal) &
								(dataFrameDetection.y < y+searchZoneY) & (dataFrameDetection.y > y-searchZoneY)] ;
					
			if dataFrameTest.empty == True :
				k = False ; numeroTrack += 1 ; 

			else :
				(rowTest,columnTest) = np.shape(dataFrameTest) ; searchZoneXReal = preciseSearchZoneX ;

				if rowTest == 1 :
					nParticle = dataFrameTest.nParticle.iat[0] 

					dataForTracking.loc[rowTracking] = [ numeroTrack, dataFrameTest.x.iat[0], dataFrameTest.y.iat[0], 
									dataFrameTest.frame.iat[0], dataFrameTest.nParticle.iat[0] ] ; 
					
					numeroFrame = dataFrameTest.frame.iat[0] ; #print('numero frame is = ', numeroFrame)
					x = dataFrameTest.x.iat[0] ; y = dataFrameTest.y.iat[0] ;
					newVelocity = dataForTracking.x.iat[rowTracking] - dataForTracking.x.iat[rowTracking-1] ; xExpected= x + newVelocity ; #print('velocity = ', velocity) 
				
					dataFrameDetection.drop(dataFrameTest.nParticle.iat[0], inplace=True) ; rowTracking += 1	;			
					
				if rowTest > 1 :
					maximalPositionX = abs(dataFrameTest['x'] - x) ; #print('maximalPositionX = ', maximalPositionX)
					indiceMax = np.argmax(maximalPositionX) ; 
						
					dataForTracking.loc[rowTracking] = [ numeroTrack, dataFrameDetection.loc[indiceMax,'x'], dataFrameDetection.loc[indiceMax,'y'], 
									dataFrameDetection.loc[indiceMax,'frame'], dataFrameDetection.loc[indiceMax,'nParticle'] ] ; 
					
					numeroFrame = dataFrameDetection.at[indiceMax,'frame'] ; #print('numro frame is = ', numeroFrame)
					x = dataFrameDetection.at[indiceMax,'x'] ; y = dataFrameDetection.at[indiceMax,'y'];
					newVelocity = dataForTracking.x.iat[rowTracking] - dataForTracking.x.iat[rowTracking-1] ; xExpected= x + newVelocity ; #print('velocity = ', velocity)
					
					dataFrameDetection.drop(dataFrameDetection.loc[indiceMax,'nParticle'], inplace=True) ; rowTracking += 1	;		


	dataFrameTrackFile = 'dataFrameTrackFile_%03d.csv'%(nDataFile+1) ; 
	dataForTracking.to_csv(dataFrameTrackFile) ;

	return()
