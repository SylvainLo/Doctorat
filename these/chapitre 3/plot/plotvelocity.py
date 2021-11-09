import numpy as np

import matplotlib.pyplot as plt

import lmfit as lm

import functionfit







###############################
####### Variable global #######
###############################
	### variable
N = 5
R0 = 2.8e-6






#################################
######### Plot velocity #########
#################################
def plotvelocity( data, fig, k, lim):
### plot preparation ###
	### plot all data
	ax = fig.add_subplot(3,2,k)

	### variable for plot
	yFit = np.linspace(0,1,101)


### Constructing data ###
	### value important in problem
	width = np.mean(data['data'][:, 2])
	print(width)


### Fit data ###
	params = fitvelocity(data, width)


### Plot data ###
	### data for plot
	vMaxIntYZ = params['V0'].value
	data['vmax'] = vMaxIntYZ * 5.86e-6 / 32

	velocity = data['velocity'][:, 1] / vMaxIntYZ

	vFitFluid = vdatasetfluid(params, yFit) / vMaxIntYZ
	vFitIntParYZ = vdataset(params, yFit) / vMaxIntYZ

	### reduction number of data
	if np.size(data['velocity'][:, 0]) > 2000 :
		limP = int ( np.floor (np.size(data['velocity'][:, 0]) / 2000) )
		velocityPlot = np.around(velocity[::limP], 4)
		yPlot = np.around(data['velocity'][::limP, 0],4)

	else :
		velocityPlot = np.around(velocity, 4)
		yPlot = np.around(data['velocity'][:, 0], 4)

	print(np.size(velocityPlot))

	### plot all curve in different graph
	ax.plot(yPlot, velocityPlot, ls='None', marker='o', markersize=1, alpha=0.2, color=data['color_velocity'])
	ax.plot(yFit, vFitIntParYZ, 'k', linewidth=1)
	ax.plot(yFit, vFitFluid, 'k:', linewidth=1)

	ax.set_xlim(0, 1) ; ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
	ax.set_ylim(0, 1.1) ; ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

	if lim is True :
		ax.text(0.95, 0.15, r'$w$ = {:.1f} $\mu $m'.format(width*1e6), horizontalalignment='right',
    verticalalignment='top')

	if k == 1  or k == 3  or k == 5:
		ax.set_ylabel(r'$v^*$')
	else :
		plt.setp(ax.get_yticklabels(), visible=False)

	if k > 4 :
		ax.set_xlabel(r'$\tilde{y}$')
	else :
		plt.setp(ax.get_xticklabels(), visible=False)


	return fig





################################
######### Fit velocity #########
################################
def fitvelocity(data, width) :

### Fit integral velocity y,z###
	### residual function
	def objective(params, position, dat) :
		residual = dat - vdataset(params, position)
		return residual.flatten()

	### Calculate Vmax
		# Creating parameter
	fitParam = lm.Parameters()
	fitParam.add('V0', np.max(data['velocity'][:, 1]),
				 min=np.min(data['velocity'][:,1]), max=np.max(data['velocity'][:, 1])*5)
	fitParam.add('radius', R0/width, vary=None)
	fitParam.add('height', data['baseData'].distance[0, 9] / width, vary=None)
	fitParam.add('n', N, vary=None)

		# Calcul fit
	result = lm.minimize(objective, fitParam, method='differential_evolution',
						  args=(data['velocity'][:, 0], data['velocity'][:, 1]))

	lm.report_fit(result.params)

	return result.params



############################################
######### function params fit/plot #########
############################################
def vdataset( params, position ):
	V0 = params['V0'].value
	h = params['height'].value
	n = params['n'].value
	r = params['radius'].value

	y = position
	return functionfit.intv3dflowyz(y, h, r, n, V0)


def vdatasetfluid( params, position ):
	V0 = params['V0'].value
	h = params['height'].value
	n = params['n'].value

	y = position
	return functionfit.vrectangularfit(y, h, V0, n)
