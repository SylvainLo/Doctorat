import matplotlib.pyplot as plt

import analysisline as al


####################################################################
### Global variable ###
	### plot latex
plt.rc( 'text', usetex=False )


	### plot
fig1 = plt.figure('deltatau')
ax1 = fig1.add_subplot(111)





####################################################################
### Main code ###
if __name__ == '__main__' :
	#################
	### File
	file1 = 'F:/bolus_concentration/network/line-30um'
	file2 = 'F:/bolus_concentration/network/network-2mm'
	file3 = 'F:/bolus_concentration/network/network-4mm'
	file4 = 'F:/bolus_concentration/network/network-8mm'

	### Analysis line
	ax1 = al.analysisline(file1, fit=True, ax=ax1, color='black', marker='o', name='l = inf')

	### Analysis intersection
	ax1 = al.analysisline( file1, fit=False, ax=ax1, color='red', marker='x', name='l = 2mm' )
	ax1 = al.analysisline( file1, fit=False, ax=ax1, color='blue', marker='^', name='l = 4mm' )
	ax1 = al.analysisline( file1, fit=False, ax=ax1, color='green', marker='s', name='l = 8mm' )


	ax1.legend( )
	plt.show()