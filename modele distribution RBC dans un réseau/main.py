import os

import numpy as np
from math import pi, cosh, sinh, sin

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.animation as anim

import trajectory as tr
import basicfunction as bf









####################
####### Main #######
####################
if __name__ == '__main__' :
	### Initialisation
	geoCh = bf.GeometryChannel()
	rbc = bf.Particle(geoCh)
	dt = 0.01


	### Simulation ###
	while bool(np.all(rbc.x > geoCh.l_total)) is False :

		### Trajectory
		rbc.position(dt=dt)

		### evenment at intersection
		if np.any(rbc.x > geoCh.lim_x[rbc.lvl_particle_branch]) :
			test = np.array(rbc.x > geoCh.lim_x[rbc.lvl_particle_branch]) * \
				   np.array(rbc.lvl_particle_branch<geoCh.lvl_max)
			rbc.update(geoCh, test)

		if np.any(np.array(rbc.x > geoCh.l_total) & rbc.finish_or_not) :
			test = np.array(rbc.x > geoCh.l_total) * rbc.finish_or_not
			rbc.time_to_finish(test)

		if rbc.counter % 20 == 0 :
			rbc.savedata()


	### y final
	rbc.final()


	### plot
		# variable
	xMin = np.min( rbc.x )
	xMax = np.max( rbc.x )
	yPlot = rbc.y1 + 2*rbc.w_particle*abs(geoCh.nbr_total_branch-rbc.nbr_particle_branch)

	nbrChannelLastLvl = 2**geoCh.lvl_max
	yWall = np.linspace(0, geoCh.w_branch[geoCh.lvl_max]*2*nbrChannelLastLvl, nbrChannelLastLvl+1)

		# figure
			# figure position (x,y)
	fig = plt.figure( 'form final front flow' )
	ax = fig.add_subplot(111)
	ax.set(xlim=(xMin, xMax), ylim=(0, np.max(yWall)))

	for i in range(0, nbrChannelLastLvl+1):
		ax.plot([xMin,xMax], [yWall[i], yWall[i]], linewidth=1, color='black')
	ax.scatter(rbc.x, yPlot, c=rbc.time_finish, s=10, cmap='inferno', alpha=0.75)

			# bar plot repartition finish time
	figHist, axHist = plt.subplots()
	n, bins, pdf = axHist.hist(rbc.time_finish, 100)


			# evolution (x,y) in time
	figAni = plt.figure()
	axAni = figAni.add_subplot(111)
	axAni.set_xlim( 0, np.max( rbc.save_x ) )
	axAni.set_ylim( 0, np.max( rbc.save_y ) )

	scat = axAni.scatter(rbc.save_x[0,:], rbc.save_y[0,:], c=rbc.time_finish, s=10, cmap='inferno', alpha=0.75)

	l, column = np.shape(rbc.save_x)
	print(column)

	def animate(ii) :
		scat.set_offsets(np.append(rbc.save_x[ii,:].reshape((-1,1)), rbc.save_y[ii,:].reshape((-1,1)), axis=1))
		return scat,

	ani = anim.FuncAnimation(figAni, animate, frames=l, interval=1)

	plt.draw()
	plt.show()