import numpy as np ;

import matplotlib.pyplot as plt ;
from mpl_toolkits.mplot3d import Axes3D ;

import scipy.interpolate as sci3 ;
import scipy.optimize as sci2 ;
import scipy.integrate as sci ;
import lmfit as lm ;



#################################
######### Mesure of tau #########
#################################

def datareorganisation(velocity, lenghtPulse, velocity0, variableFile) :
	(line,column) = np.shape(lenghtPulse) ;
	print('max velocity = ', velocity[:,1]) ; print('x = ', variableFile.distance[:,1]) ; print('lenght pulse = ', lenghtPulse[:,1]) ;


	# Calcul lenght pulse without initial pulse
	lenghtPulseNormalize = np.zeros((1,column)) ; 
	X1 = np.array([0]) ; 
	V0 = np.array([velocity0[0]]) ; 
	V = np.zeros((1,column)) ; V[0,:] = velocity[0,:] ; 

	print(np.shape(variableFile.distance))

	for i in range(line) :
		#print(i)
		if variableFile.distance[i,1] == 0 :
			lenghtPulse0 = lenghtPulse[i,:] ;
		else :
			A = lenghtPulse[i,:] - lenghtPulse0 ; A = np.reshape(A, (1,column)) ; lenghtPulseNormalize = np.append(lenghtPulseNormalize, A, axis=0) ; 
			B = np.reshape(velocity[i,:],(1,column)) ; V = np.append(V, B, axis=0) ;
			X1 = np.append(X1, variableFile.distance[i,1]) ; 
			V0 = np.append(V0, velocity0[i]) ; 


	# Sorting array
	(line,column) = np.shape(lenghtPulseNormalize) ;

	sortingArray = np.argsort(X1) ; 
	X1 = np.array(X1)[sortingArray] ; V0 = np.array(V0)[sortingArray] ;

	X1global = np.copy(X1) ; V0global = np.copy(V0) ; Vglobal = np.copy(V) ;

	for j in range(column) :
		lenghtPulseNormalize[:,j] = lenghtPulseNormalize[:,j][sortingArray] ;
		Vglobal[:,j] = Vglobal[:,j][sortingArray] ;
		if j >= 1 :
			X1global = np.append(X1global, X1, axis=0) ;
			V0global = np.append(V0global, V0, axis=0) ;
			

	# Calcul in meter and sec
	tauExp = lenghtPulseNormalize / variableFile.distance[i,4]  ; #print('tauExp = ', tauExp)
	
	Vglobal = Vglobal * variableFile.distance[0,4] * variableFile.liaison_pix_to_um ; 
	
	X1global = np.reshape(X1global, (column,line)) ;  X1global = np.transpose(X1global) ; 
	X1global = X1global * 1e-6 ; 
	
	V0global = np.reshape(V0global, (column,line)) ; V0global = np.transpose(V0global) ; 
	V0global = V0global * variableFile.distance[0,4] * variableFile.liaison_pix_to_um ; 
	
	#print('V0global = ', V0global) ; print('X1global = ', X1global) ; print('Vglobal = ', Vglobal) ;

	np.savetxt('X1global.txt', X1global, fmt='%1.4e',delimiter='    ')
	np.savetxt('V0global.txt', V0global, fmt='%1.4e',delimiter='    ')
	np.savetxt('Vglobal.txt', Vglobal, fmt='%1.4e',delimiter='    ')
	np.savetxt('tauExp.txt', tauExp, fmt='%1.4e',delimiter='    ')

	return()







##############################
######### Fit of tau #########
##############################

def measuretau(Vglobal, tauExp, V0global, X1global, variableFile, dataForGlobalFit) :
	### Constant for fit ###
		# For vector construction
	(line, column) = np.shape(tauExp) ;

		# Basic variable
	R0 = variableFile.rayon_globule_rouge ; width = variableFile.distance[0,3] ; y0 = 0 ; 

		# Experimental variable
	V0cte = np.mean(abs(V0global)) ; print('V0cte = ', V0cte) ; 

	yIni = np.zeros(line) ; VonV0 = np.zeros(line) ; VonV0 = ((V0global[0,:]-Vglobal[0,:]) / V0global[0,:]) ;
	yIni = width * (1-np.sqrt(VonV0)) ;
	print('yIni = ', yIni) ;

	dataForGlobalFit["tauExp{0}".format(variableFile.n_files)] = tauExp[:,0] ;
	dataForGlobalFit["V0{0}".format(variableFile.n_files)] = V0cte ;
	dataForGlobalFit["X{0}".format(variableFile.n_files)] = X1global[:,0] ;
	dataForGlobalFit["V0global{0}".format(variableFile.n_files)] = V0global[:,0] ;



	## Different basic plot
	#plt.figure(70)
	#plt.plot(X1global[:,0], tauExp[:,0], 'o', color=variableFile.color_file[variableFile.n_files]) ;
	#plt.title('$\tau (x)$')

	#plt.figure(71)
	#plt.plot(X1global[1:,0], (X1global[1:,0]*V0cte)/(V0cte*tauExp[1:,0]+X1global[1:,0]), 'o', color=variableFile.color_file[variableFile.n_files]) ;
	#plt.title('<V>')

	#plt.figure(72)
	#plt.plot(X1global[:,0], (tauExp[:,0]*V0global[:,0]), 'o', color=variableFile.color_file[variableFile.n_files])
	#plt.plot(X1global[:,0], (tauExp[:,0]*V0cte), '+', color=variableFile.color_file[variableFile.n_files])

	#plt.figure(77)
	#plt.plot(X1global[1:,0], (X1global[1:,0]*V0global[1:,0])/(V0global[1:,0]*tauExp[1:,0]+X1global[1:,0]), 'o', color=variableFile.color_file[variableFile.n_files]) ;
	#plt.title('<V> with V0global')

	#plt.figure(78)
	#plt.plot((X1global[1:,0])/(V0cte*tauExp[1:,0]+X1global[1:,0]), tauExp[1:,0],  'o', color=variableFile.color_file[variableFile.n_files]) ;
	#plt.xlabel('<V>/V0') ; plt.ylabel('tau')

		
	#### Formule fit ###
	def tau(x, delta, xi, yIni) :
		### first method 
			# ode
		def dydxfit(y, x, xi, delta) :
			return ( xi * R0**(delta+1) * (width-y) ) / ( y * (width-y/2) * ((y-y0)**delta) ) 
		
		# calcul position y
		y = sci.odeint(dydxfit, yIni, x, args=(xi,delta)) ; 
		yShape = np.size(y); y = np.reshape(y,yShape) ; 
		# calcul position x
		V = ( 2*V0cte/(width**2) ) * ( y*width - (y**2)/2 ) ; 
		intV = np.empty(yShape) ; intV[0]=0 ;  
		intV[1:] = sci.cumtrapz(V,x) ;

		# calcul tau 
		tauFit = np.empty(yShape) ; tauFit[0] = 0 ;
		tauFit[1:] =  x[1:] * ( x[1:]/intV[1:] - 1/V0cte ) ;  

		return tauFit




	#### Brute method ###
	#	### Residu by curve
	#nDelta = 25 ; delta0 = np.linspace(0,10,nDelta) ;
	#nXi = 100 ; xi0 = np.linspace(0.000001,2.5, nXi) ;
	#yIni0 = 10e-6 ;

	#X,Y = np.meshgrid(delta0, xi0, indexing='ij', sparse=False) ; #print('size X = ', np.shape(X))
	#residu = np.zeros((nDelta, nXi)) ;

	#for i in range(nDelta) :
	#	#print(i)
	#	for j in range(nXi) :
	#		residu[i,j] = np.sqrt(np.sum((tauExp[:,0] - tau(X1global[:,0],delta0[i],xi0[j],yIni0))**2, axis=0 )) ;

	#positionMin = np.argmin(residu) ;
	#positionMin = np.unravel_index(positionMin, (nDelta, nXi))

	#	### Curve min
	#minResiduDelta = np.array( [ delta0]) ; minResiduXi = np.array([ xi0[np.argmin(residu, axis=1)] ]) ;

	#	### Fit curve min
	#def func(x, a, b, c):
	#	return a*np.exp(b*x) + c
	#popt, pcov = sci2.curve_fit(func, minResiduDelta[0,:], minResiduXi[0,:], 
	#						bounds=([0, 0, 0],[10,10,10])) ; print(popt)

	#plt.figure(80+variableFile.n_files)
	#plt.pcolormesh(X,Y,residu, cmap='coolwarm', vmax=np.min(residu)*10) ;
	#plt.plot(delta0[positionMin[0]], xi0[positionMin[1]], marker='+', markersize=25, color='chartreuse')
	#plt.plot(minResiduDelta, minResiduXi, color='green', marker='o', markersize=3, linestyle='none') ;
	#plt.plot(minResiduDelta[0,:], func(minResiduDelta[0,:], popt[0], popt[1], popt[2]), linestyle=':', color='chartreuse',
	#	 label=('fit : ', "%.3E"%(popt[0]), '* exp(x *', "%.3E"%(popt[1]),
	#				') + ', "%.3E"%(popt[2]))) ;
	#plt.legend()
	#plt.colorbar(aspect=15)



	#fig = plt.figure(160+variableFile.n_files)
	#ax = fig.add_subplot(111, projection='3d')
	#ax.plot_surface(X, Y, residu, cmap='coolwarm', vmax=5)




	#### Curve fit by minimisation ###
	for o in range(column) :
	#	### fit with all parameter free ###
	#	popt, pcov = sci2.curve_fit(tau, X1global[:,o], tauExp[:,o], 
	#						  p0=(1, 0.001, yIni[o]), bounds=([0, 0, y0+0.01e-6],[5, 5, width]) ) ;
	#	fittingCurve = tau(X1global[:,o], popt[0], popt[1], popt[2]) ;
	#	print('         delta = ', popt[0], ', xi = ', popt[1], ', yIni = ', popt[2]) ;

	#		# Plot
	#	plt.figure(90+variableFile.n_files) ;
	#	plt.plot( X1global[:, o], tauExp[:, o], 'o', color=variableFile.color_file[o] )  ;
	#	plt.plot( X1global[:, o], fittingCurve, ':', color=variableFile.color_file[o] ,label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(popt[0],popt[1],popt[2])) ) ;
	#	plt.legend()

	#	plt.figure(200)
	#	plt.plot( X1global[:, o], tauExp[:, o], 'o', color=variableFile.color_file_video[o+6*variableFile.n_files] )  ;
	#	plt.plot( X1global[:, o], fittingCurve, ':', color=variableFile.color_file_video[o+6*variableFile.n_files] ,label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(popt[0],popt[1],popt[2])) ) ;
	#	plt.legend()


	#		# utilisation des resultat pour tracer la vitesse de migration
	#	if o == 0 :
	#			# Test curve x
	#				# find x and tau
	#		x = np.linspace(X1global[0,0], X1global[-1,0], 1000) ;
	#		fittingCurve2 = tau(x, popt[0], popt[1], popt[2]) ;

	#				# dtaudxFit
	#		dtaudxFit2 = np.gradient(fittingCurve2, x, edge_order=2) ;

	#				# yFromTau
	#		yFromTauFit2 = width * (1 - np.sqrt( ((V0cte*fittingCurve2)**2 + x**2*V0cte*dtaudxFit2) / ((V0cte*fittingCurve2+x)**2) )) ;

	#				# dydtFit
	#		dydxFit2 = np.gradient(yFromTauFit2, x, edge_order=2)

	#			# Plot
	#				# dtaudxFit
	#		plt.figure(136)
	#		plt.plot(x, dtaudxFit2, '--', color=variableFile.color_file[variableFile.n_files])
	#		plt.xlabel('x') ; plt.ylabel('dtau/dx') ;

	#				# yFromTau
	#		plt.figure(137)
	#		plt.plot(x, yFromTauFit2, '--', color=variableFile.color_file[variableFile.n_files])
	#		plt.xlabel('x') ; plt.ylabel('y(x,tau,dtau/dx)') ;
	#		plt.legend()

	#				# dydtFit
	#		plt.figure(138)
	#		plt.plot(yFromTauFit2, dydxFit2, '-', color=variableFile.color_file[variableFile.n_files], label=('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(popt[0],popt[1],popt[2])) )
	#		plt.xlabel('y(x,tau,dtau/dx)') ; plt.ylabel('dy/dx') ;
	#		plt.legend()


	#			# Fit curve min
	#		def dydxfit(y, delta, xi):
	#			return ( xi * R0**(delta+1) * (width-y) ) / ( y * (width-y/2) * ((y-y0)**delta) )
	#		poptFit, pcov = sci2.curve_fit(dydxfit, yFromTauFit2[2:], dydxFit2[2:], bounds=([0, 0],[10,10])) ; 
		
	#		print('delta dydxfit = ', poptFit[0], ', xi dydxfit = ', poptFit[1]) ;
	#		#plt.figure(138)
	#		#plt.plot(yFromTauFit2, dydxfit(yFromTauFit2, poptFit[0], poptFit[1] ), ':', color=variableFile.color_file[variableFile.n_files] ,label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(poptFit[0],poptFit[1],poptFit[2])) ) ;
	#		#plt.legend()




		### Fit parameter with constrained parameter
		popt2, pcov = sci2.curve_fit(tau, X1global[:,o], tauExp[:,o], p0=(1, 0.001, yIni[o]),
							   bounds=([0.99999, 0, y0+0.01e-6],[1.00001, 5, width]) );
		fittingCurveP = tau(X1global[:,o], popt2[0], popt2[1], popt2[2]) ;
		print('         delta = ', popt2[0], ', xi = ', popt2[1], ', yIni = ', popt2[2]) ;

			# Plot
		plt.figure(90+variableFile.n_files) ;
		plt.plot( X1global[:, o], tauExp[:, o], 'o', color=variableFile.color_file[o] )  ;
		plt.plot( X1global[:, o], fittingCurveP, ':', color=variableFile.color_file[o] ,label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(popt2[0],popt2[1],popt2[2])) ) ;
		plt.legend()

		#plt.figure(200)
		#plt.plot( X1global[:, o], tauExp[:, o], 'o', color=variableFile.color_file_video[o+6*variableFile.n_files] )  ;
		#plt.plot( X1global[:, o], fittingCurveP, ':', color=variableFile.color_file_video[o+6*variableFile.n_files] ,label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(popt2[0],popt2[1],popt2[2])) ) ;
		#plt.legend()

			# utilisation des resultat pour tracer la vitesse de migration
		if o == 0 :

			plt.figure(200)
			plt.title('$\Phi = 1 \%, W = 20 \mu m, \delta = 1 $', size=20)
			plt.plot( X1global[:, o], tauExp[:, o], 'o', color=variableFile.color_file[variableFile.n_files], label='Experimental point {}'.format(variableFile.n_files) )  ;
			plt.plot( X1global[:, o], fittingCurveP, ':', color=variableFile.color_file[variableFile.n_files] ,label = ('fit : $ \\xi $ ={:.2E}, yIni ={:.2E} '.format(popt2[1],popt2[2])) ) ;	
			plt.xlabel('x (m)', size=17) ; plt.ylabel('$\\tau (s) $ ', size=17) ;
			plt.tick_params(axis='x', labelsize=15) ; plt.tick_params(axis='y', labelsize=15) ;

			plt.figure(202)
			plt.plot(variableFile.viscosity[variableFile.n_files], popt2[1], 'o', markerSize=8, color=variableFile.color_file[variableFile.n_files])
			plt.xlabel('$ \\nu  (mPa.s) $', size=17) ; plt.ylabel('$ \\xi $', size=17) ;
			plt.ticklabel_format(style='sci', axis=('y'), scilimits=(2,0)) ;
			plt.tick_params(axis='x', labelsize=15) ; plt.tick_params(axis='y', labelsize=15) ;


				# Test curve x
					# find x and tau
			x = np.linspace(X1global[0,0], X1global[-1,0], 1000) ;
			fittingCurve2 = tau(x, popt2[0], popt2[1], popt2[2]) ;

					# dtaudxFit
			dtaudxFit2 = np.gradient(fittingCurve2, x, edge_order=2) ;

					# yFromTau
			yFromTauFit2 = width * (1 - np.sqrt( ((V0cte*fittingCurve2)**2 + x**2*V0cte*dtaudxFit2) / ((V0cte*fittingCurve2+x)**2) )) ;

					# dydtFit
			dydxFit2 = np.gradient(yFromTauFit2, x, edge_order=2)

				# Plot
					# dtaudxFit
			plt.figure(136)
			plt.plot(x, dtaudxFit2, '--', color=variableFile.color_file[variableFile.n_files])
			plt.xlabel('x') ; plt.ylabel('dtau/dx') ;

					# yFromTau
			plt.figure(137)
			plt.plot(x, yFromTauFit2, '--', color=variableFile.color_file[variableFile.n_files])
			plt.xlabel('x') ; plt.ylabel('y(x,tau,dtau/dx)') ;
			plt.legend()

					# dydtFit
			plt.figure(201)
			plt.plot(yFromTauFit2, dydxFit2, '-', color=variableFile.color_file[variableFile.n_files], label=('fit : $ \\xi $={:.2E}, yIni ={:.2E}'.format(popt2[1],popt2[2])) )
			plt.xlabel('y') ; plt.ylabel('dy/dx') ;
			plt.legend()


			#	# Fit curve min
			#def dydxfit(y, delta, xi):
			#	return ( xi * R0**(delta+1) * (width-y) ) / ( y * (width-y/2) * ((y-y0)**delta) )
			#poptFit, pcov = sci2.curve_fit(dydxfit, yFromTauFit2[2:], dydxFit2[2:], bounds=([0, 0],[10,10])) ; 
		
			#print('delta dydxfit = ', poptFit[0], ', xi dydxfit = ', poptFit[1]) ;
			#plt.figure(138)
			#plt.plot(yFromTauFit2, dydxfit(yFromTauFit2, poptFit[0], poptFit[1] ), ':', color=variableFile.color_file[variableFile.n_files] ,label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(poptFit[0],poptFit[1],poptFit[2])) ) ;
			#plt.legend()



	#### Curve fit by differential evolution ###
	#	### Mise en place function of parameter

	#print('   We begin the analysis with lmfit')

	#		# function
	#def tau_dataset(params, i, position1) :
	#	delta = params['delta'].value
	#	xi = params['xi'].value
	#	yIniFit = params['yIni_%i'%(i)].value ;
	#	xX = position1[:,i] ;
	#	return tau(xX, delta, xi, yIniFit)

	#		# residual function
	#def objective(params, position, data) :
	#	(lineObj, rowObj) = np.shape(data) ;
	#	resid = np.ones((lineObj, rowObj))	; 
	#	for i in range(rowObj) :
	#		resid[:,i] = data[:,i] - tau_dataset(params, i, position) ;
	#	return resid.flatten()


	#		### Calcul delta, xi et y0 optimal with the method differential_evolution'
	#			# Mise en place parameter
	#fit_params = lm.Parameters()
	#for iy in range(column) :
	#	fit_params.add( 'delta', 0, min=0, max=5)
	#	fit_params.add( 'xi', 1, min=1e-5, max=1)
	#	fit_params.add( 'yIni_%i'%(iy) , width, min=R0+1e-6, max=width)

	#			# Calcul fit
	#result = lm.minimize(objective, fit_params, method='differential_evolution', args=(X1global, tauExp))
	#lm.report_fit(result.params)

	#factorTauFinal = (1 / (V0cte*2*result.params['xi'].value*(1+result.params['delta'].value)*(2+result.params['delta'].value)*R0**(result.params['delta'].value+1)) ) ;
	#firstMemberTauFinal = (width-R0)**(2+result.params['delta'].value)
	#secondMemberTauFinal = ((result.params['yIni_0'].value-R0)**(1+result.params['delta'].value)) * ((width-result.params['yIni_0'].value)*(1+result.params['delta'].value)+width-R0) ;
	#print('tau final would be : ', factorTauFinal*(firstMemberTauFinal-secondMemberTauFinal))

	#			# Plot
	#for i in range(column):
	#	y_fit = tau_dataset(result.params, i, X1global)
	#	plt.figure(100+variableFile.n_files)
	#	plt.plot(X1global[:, i], tauExp[:, i], 'o', color=variableFile.color_file[i]) 
	#	plt.plot(X1global[:, i], y_fit, '-', color=variableFile.color_file[i], label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(result.params['delta'].value,result.params['xi'].value,result.params['yIni_{}'.format(i)].value)))


	#		### Calcul delta, xi et y0 optimal with the method minimization'
	#			# Mise en place parameter
	#fit_params2 = lm.Parameters()
	#for iy in range(column) :
	#	fit_params2.add( 'delta', 0, min=1, max=2)
	#	fit_params2.add( 'xi', 1, min=1e-5, max=1)
	#	fit_params2.add( 'yIni_%i'%(iy) , yIni[iy], min=yIni[iy]-1e-9, max=yIni[iy]+1e-9)

	#			# Calcul fit
	#result2 = lm.minimize(objective, fit_params2, method='differential_evolution', args=(X1global, tauExp))
	#lm.report_fit(result2.params)

	#			# Plot
	#for i in range(column):
	#	y_fit = tau_dataset(result2.params, i, X1global)
	#	plt.plot(X1global[:, i], y_fit, ':', color=variableFile.color_file[i], label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(result2.params['delta'].value,result2.params['xi'].value,result2.params['yIni_{}'.format(i)].value)))
	#	plt.legend()


	return(dataForGlobalFit)







def measureX(Vglobal, tauExp, V0global, X1global, variableFile, dataForGlobalFit) :
	### Constant for fit ###
		# For vector construction
	(line, column) = np.shape(tauExp) ;

		# Basic variable
	R0 = variableFile.rayon_globule_rouge ; width = variableFile.distance[0,3] ; y0 = 0 ; 

		# Experimental variable
	print('Vglobal = ', Vglobal)

	V0cte = np.mean(abs(V0global)) ; print('V0cte = ', V0cte) ; 

	y = np.zeros((line, column)) ; VonV0 = np.zeros((line, column)) ; VonV0 = ((V0global-Vglobal) / V0global) ;
	y = width * (1-np.sqrt(VonV0)) ;
	print('y = ', y) ;

	dataForGlobalFit["y{0}".format(variableFile.n_files)] = y[:,0] ;

		# experimental data
	Xon2 = 2*V0global*tauExp ;

		# Different basic plot
	plt.figure(180+variableFile.n_files)
	for i in range(column) :
		plt.plot(X1global[:,0], y[:,i], 'o', color=variableFile.color_file_video[7*i], label='{}'.format(variableFile.limit_velocity[i]))
	plt.legend()

	plt.figure(170)
	plt.plot(X1global[:,0], 2*V0global[:,0]*tauExp[:,0], 'o', color=variableFile.color_file[variableFile.n_files])
	plt.title('X/2 (lenght of the pulse)')

	plt.figure(171)
	plt.plot(X1global[:,0], y[:,0], 'o', color=variableFile.color_file[variableFile.n_files], label='{}'.format(variableFile.n_files))
	plt.title('position y')
	plt.legend()

	plt.figure(172)
	plt.plot(2*V0global[:,0]*tauExp[:,0], y[:,0], 'o', color=variableFile.color_file[variableFile.n_files])
	plt.title('y en fonction de X/2')


	#### Formule fit ###
	def X(y, delta, xi, yIni) :
				
		def A(yvalue, delta, xi) :
			return((yvalue-R0)**(1+delta) * ((width-yvalue)*(1+delta)+width-R0))	

		B = 1 / (xi * (1+delta) * (2+delta) * R0**(delta+1)) ;
		Acte = A(yIni,delta,xi) ;
		Avar = A(y,delta,xi) ;

		return (B*(Avar-Acte))

	
	print('   We begin the analysis with lmfit')

	### function ###
	def X_dataset(params, i, position1) :
		delta = params['delta'].value
		xi = params['xi'].value
		yIniFit = params['yIni_%i'%(i)].value ;
		yy = position1[:,i] ;
		return X(yy, delta, xi, yIniFit)

		# residual function
	def objective(params, position, data) :
		(lineObj, rowObj) = np.shape(data) ;
		resid = np.ones((lineObj, rowObj))	; 
		for i in range(rowObj) :
			resid[:,i] = data[:,i] - X_dataset(params, i, position) ;
		return resid.flatten()


	### Calcul delta, xi et yIni optimal with the method differential_evolution'
		# Mise en place parameter
	fitParamsX = lm.Parameters()
	for iy in range(column) :
		fitParamsX.add( 'delta', 0, min=0, max=5)
		fitParamsX.add( 'xi', 1, min=1e-5, max=1)
		fitParamsX.add( 'yIni_%i'%(iy) , y[0,iy], min=R0+1e-9, max=width)

		# Calcul fit
	result = lm.minimize(objective, fitParamsX, method='differential_evolution', args=(y, Xon2))
	lm.report_fit(result.params)

		# Plot fit
	for i in range(column):
		Xfit = X_dataset(result.params, i, y)
		plt.figure(400+variableFile.n_files)
		plt.plot(X1global[:, i], Xon2[:, i], 'o', color=variableFile.color_file[i]) 
		plt.plot(X1global[:, i], Xfit, '-', color=variableFile.color_file[i], label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(result.params['delta'].value,result.params['xi'].value,result.params['yIni_{}'.format(i)].value)))
		plt.legend()

		# Plot fit y
			#def function
	def dydxfit(y, x, xi, delta) :
		return ( xi * R0**(delta+1) * (width-y) ) / ( y * (width-y/2) * ((y-y0)**delta) ) 
		 
	for i in range(column):
		yFit = sci.odeint(dydxfit, result.params['yIni_{}'.format(i)].value, X1global[:,0], args=(result.params['xi'].value, result.params['delta'].value)) ;
		plt.figure(500+variableFile.n_files)
		plt.plot(X1global[:, i], y[:, i], 'o', color=variableFile.color_file[i]) 
		plt.plot(X1global[:, i], yFit, '-', color=variableFile.color_file[i], label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(result.params['delta'].value,result.params['xi'].value,result.params['yIni_{}'.format(i)].value)))
		plt.legend()


	return(dataForGlobalFit)












###############################
######### Fit of tau ##########
###############################


def fitalldata(dataForGlobalFit, dictionnaryForVariable) :

	#### Formule fit ###
		### Constante
	def tau(x, delta, xi, yIni, i) : 
		R0 = dictionnaryForVariable['variableForFile{}'.format(i)].rayon_globule_rouge ;
		width = dictionnaryForVariable['variableForFile{}'.format(i)].distance[0,3] ;
		V0 = dataForGlobalFit["V0{}".format(i)] ;
		y0 = 0

		### first method 
			# ode
		def dydxfit(y, x, xi, delta) :
			return ( xi * R0**(delta+1) * (width-y) ) / ( y * (width-y/2) * ((y-y0)**delta) ) 
		
		# calcul position y
		y = sci.odeint(dydxfit, yIni, x, args=(xi,delta)) ; 
		yShape = np.size(y); y = np.reshape(y,yShape) ; 
		# calcul position x
		V = ( 2*V0/(width**2) ) * ( y*width - (y**2)/2 ) ; 
		intV = np.empty(yShape) ; intV[0]=0 ;  
		intV[1:] = sci.cumtrapz(V,x) ;

		# calcul tau 
		tauFit =  x * ( x/intV - 1/V0 ) ; tauFit[0] = 0 ; 
		#print('delta = {}, xi = {}'.format(delta,xi))

		return tauFit



	### Preparation fit  ###
		# function
	def tau_dataset(params, i, position1) :
		delta = params['delta'].value
		xi = params['xi'].value
		yIniFit = params['yIni_%i'%(i)].value ;
		xX = position1 ;
		return tau(xX, delta, xi, yIniFit, i)

		# residual function
	def objective(params, position, data) :
		resid = np.array([]) ;
		lineObj = len(dictionnaryForVariable) ;
		for i in range(lineObj) :
			resid = np.append(resid,  (dataForGlobalFit["tauExp{0}".format(i)] - tau_dataset(params, i, dataForGlobalFit["X{0}".format(i)])) ) ;
		#print('residu = ', np.sum(np.abs(resid)))
		return resid


	### Calcul delta, xi et y0 optimal with the method differential_evolution'
		# Mise en place paramete
	fit_params = lm.Parameters()
	for iy in range( len(dictionnaryForVariable) ) :
		fit_params.add( 'delta', 1, vary=False)
		fit_params.add( 'xi', 1, min = 1e-5, max = 1)
		fit_params.add( 'yIni_%i'%(iy) , dictionnaryForVariable["variableForFile{0}".format(iy)].distance[0,3], 
				 min=dictionnaryForVariable["variableForFile{0}".format(iy)].rayon_globule_rouge+1e-6, 
				 max=dictionnaryForVariable["variableForFile{0}".format(iy)].distance[0,3])

		# Calcul fit
	result = lm.minimize(objective, fit_params, method='differential_evolution', args=(dataForGlobalFit, dictionnaryForVariable))
	lm.report_fit(result.params)

		# Plot
	for i in range(len(dictionnaryForVariable)):
		y_fit = tau_dataset(result.params, i, dataForGlobalFit["X{0}".format(i)])
		plt.figure(300)
		plt.plot( dataForGlobalFit["X{0}".format(i)], dataForGlobalFit["tauExp{0}".format(i)], 'o', color=dictionnaryForVariable['variableForFile0'].color_file[i] ) 
		plt.plot( dataForGlobalFit["X{0}".format(i)], y_fit, '-', color=dictionnaryForVariable['variableForFile0'].color_file[i], label = ('fit : delta={:.2E}, xi={:.2E}, yIni={:.2E}'.format(result.params['delta'].value,result.params['xi'].value,result.params['yIni_{}'.format(i)].value)) )
		plt.legend()


