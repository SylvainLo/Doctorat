import os

import matplotlib.pyplot as plt
import matplotlib as mpl

import multiprocessing

import pltc
import pltprofile
import xialongbolus


####################################################################
### Global variable ###
	### number processor
num_cores = multiprocessing.cpu_count( )

	### plot latex
font = {'size':10}
plt.rc( 'font', **font )
rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(rc_params)

	### figure
# color
c = ['r', 'b', 'g']
sb = ['o', 'v', 's']
ls = ['--', ':', '-.']

	### plot
fig1 = plt.figure( 'concentration' )

fig2 = plt.figure( 'lambda' )

fig3 = plt.figure( 'density' )



####################################################################
### Main code ###
if __name__ == '__main__' :
	##################
	### Choice directory
	nameHDD = 'G:/'
	nameDir = os.path.join( nameHDD, 'bolus_concentration/' )
	while os.path.isdir( nameDir ) is False :
		print( '   Error : your directory does not exist !' )
		nameDir = input( 'Directory of your data (E:/.../) : ' )
	os.chdir( nameDir )

	# pltc.plotconcentration(fig1, nameDir, c, sb, ls)
	# os.chdir( nameDir )
	# fig1.savefig( 'bolus_concentration.svg', format='svg' )
	# fig1.savefig( 'bolus_concentration.pdf', format='pdf' )
	# #
	# pltc.plotlambda(fig2, nameDir, c, sb, ls)
	# os.chdir( nameDir )
	# fig2.savefig( 'bolus_lambda.svg', format='svg' )
	# fig2.savefig( 'bolus_lambda.pdf', format='pdf' )
	# #
	# pltc.plotdensity(fig3, nameDir, c, sb, ls)
	# os.chdir( nameDir )
	# fig3.savefig( 'bolus_density.svg', format='svg' )
	# fig3.savefig( 'bolus_density.pdf', format='pdf' )

	# fig4 = pltprofile.plotprofile(nameDir, c)
	# os.chdir( nameDir )
	# fig4.savefig( 'profile_bolus.svg', format='svg' )
	# fig4.savefig( 'profile_bolus.pdf', format='pdf' )

	fig5 = xialongbolus.test(nameDir, c, sb)

	fig6 = xialongbolus.onlyone(nameDir, c, sb)

	plt.show()