import os

import numpy as np
from math import pi, cosh, sinh, sin
import matplotlib.pyplot as plt

import trajectory as tr







class GeometryChannel :
	def __init__(self):
	### number of branches
		self.lvl_max = 4 #int( input( 'number of branches ?' ) )

	### number total of branch, their name and their properties
		# total number
		self.nbr_total_branch = int( 2**(self.lvl_max+1) - 1) # only one straight channel = 0

		# number of the branch
		self.name_branch = np.linspace( 1, self.nbr_total_branch, self.nbr_total_branch, dtype=np.int64 )

		# level of the branch
		self.value_branch = np.ones( self.nbr_total_branch, dtype=np.int64 )

	### connectivity table
		if self.lvl_max == 0 :
			self.connectivity = []

		elif self.lvl_max == 1 :
			self.connectivity = [[1, 2, 3]]

		else :
			self.connectivity = [[1, 2, 3]]
			for i in range( 1, int(self.nbr_total_branch - self.lvl_max**2) ) :
				self.connectivity.append( [self.connectivity[i-1][0]+1, self.connectivity[i-1][1]+2,
										   self.connectivity[i-1][2]+2] )

	### properties of each branches
		# initialisation of the properties of each level of intersection
		self.height = 15e-6 / 2  # float( input( 'height in m in the branch' ) ) / 2
		self.w_branch = np.ones( self.lvl_max+1, dtype=np.float64 )
		self.l_branch = np.ones( self.lvl_max+1, dtype=np.float64 )
		self.v_branch = np.ones( self.lvl_max+1, dtype=np.float64 )
		self.v_max = 1.75e-3 #float( input( 'maximal velocity in the first branch in m.s-1 ?' ) )

		# value in each intersection
		for i in range( self.lvl_max+1 ) :
			self.value_branch[2**i-1 :2**(i+1)-1] = i
			self.w_branch[i] = 16e-6/2 #float( input( 'width in m in the {} branch'.format( i+1 ) ) ) / 2
			self.l_branch[i] = 1.26e-3 #float( input( 'length in m in the {} branch'.format( i+1 ) ) )
			self.v_branch[i] = self.v_max * (self.w_branch[0] * self.height) / \
							   (self.w_branch[i] * self.height * 2**i)

		# other value
		self.l_total = np.sum( self.l_branch )
		self.lim_x = np.cumsum( self.l_branch )


	### value for plot






class Particle :
### initialisation particles ###
	def __init__(self, geo):
	### particle properties
		self.nbr_particle = 1000 #int( input( 'number of particles ?' ) )
		self.size_particle = 2.8e-6
		self.lim_wall = 1e-6
		self.xi = 0 #1.2e-2
		self.delta = 1.2

	### position
		# initialisation
		self.x = np.random.normal(-5e-3, 5e-3/3, self.nbr_particle)#np.zeros( self.nbr_particle, dtype=np.float64 )
		self.x[self.x > 0] = 0
		self.y1 = np.random.random(self.nbr_particle) * 2 * (geo.w_branch[0] - self.lim_wall) +  self.lim_wall#np.linspace( self.lim_wall, geo.w_branch[0]*2-self.lim_wall, self.nbr_particle, dtype=np.float64 )
		self.y2 = np.zeros(self.nbr_particle, dtype=np.float64)

		self.time = float(0)
		self.finish_or_not = np.ones( self.nbr_particle, dtype=np.bool_ )
		self.time_finish = np.zeros( self.nbr_particle, dtype=np.float64 )

		self.counter = int(0)

		plt.figure('test')
		plt.plot(self.y1, self.x, 'o')

		# particle properties depending their branches
		self.lvl_particle_branch = np.zeros( self.nbr_particle, dtype=np.int64 )
		self.nbr_particle_branch = np.ones( self.nbr_particle, dtype=np.int64 )
		self.w_particle = np.zeros( self.nbr_particle )+geo.w_branch[0] # half-width
		self.v_particle = np.zeros( self.nbr_particle )+geo.v_branch[0]

		# adaptation to non symmetrical equation of migration
		self.particle_top = np.zeros( self.nbr_particle, dtype=np.bool_ )
		self.particle_top[self.y1 > self.w_particle] = True
		self.y1[self.particle_top] =  2*self.w_particle[self.particle_top]  - self.y1[self.particle_top]


	### constant for integration
		# independent of geometry
		self.b1 = pi / (2 * geo.height)
		self.b3 = 3 * pi / (2 * geo.height)
		self.b5 = 5 * pi / (2 * geo.height)

		self.A1 = np.sin( self.b1 * self.size_particle )
		self.A3 = - 1 / 27 * np.sin( self.b3 * self.size_particle )
		self.A5 = 1 / 125 * np.sin( self.b5 * self.size_particle )

		# dependent of geometry
			# prefactor
		self.v0 = self.v_particle / ( 1 * (1-1/np.cosh(self.b1*self.w_particle)) -
									  1/9 * (1-1/np.cosh(self.b3*self.w_particle)) +
									  1/25 * (1-1/np.cosh(self.b5*self.w_particle)) )

			# vy = mu_U/y**delta * sum(mu_An * sinh(bn*(y-w))
		self.mu_U = self.xi * self.size_particle**(self.delta+1) * self.v0
		self.mu_A1 = -1 * self.b1 / np.cosh(self.b1*self.w_particle)
		self.mu_A3 = 1/9 * self.b3 / np.cosh(self.b3*self.w_particle)
		self.mu_A5 = -1/25 * self.b5 / np.cosh(self.b5*self.w_particle)

			#vx = vx_U * sum(vx_An * (2*r - vx_Bn*(cosh(bn*(y1-w))+cosh(bc*(y2-w)) ) )
		self.vx_U = self.v0 * geo.height / (2 * pi * self.size_particle**2)
		self.vx_A1 = 1 * np.sin(self.b1 * self.size_particle)
		self.vx_A3 = -1/27 * np.sin(self.b3 * self.size_particle)
		self.vx_A5 = 1/125 * np.sin(self.b5 * self.size_particle)
		self.vx_B1 = np.sinh(self.b1 * self.size_particle ) / (self.b1 * np.cosh( self.b1 * self.w_particle ))
		self.vx_B3 = np.sinh(self.b3 * self.size_particle ) / (self.b3 * np.cosh( self.b3 * self.w_particle ))
		self.vx_B5 = np.sinh(self.b5 * self.size_particle ) / (self.b5 * np.cosh( self.b5 * self.w_particle ))

	### vector save
		self.save_x = np.zeros((1, self.nbr_particle))
		self.save_y = self.y1.reshape((1, self.nbr_particle))
		self.save_lvl = np.zeros((1, self.nbr_particle))
		self.save_nbr_channel = np.ones((1, self.nbr_particle))


### evolution position at each time step ###
	def position( self, dt ):
	### evolution y
		k1 = dt * tr.vy(self.y1, self)
		k2 = dt * tr.vy(self.y1+0.5*k1, self)
		k3 = dt * tr.vy(self.y1+0.5*k2, self)
		k4 = dt * tr.vy(self.y1+k3, self)
		dy = (k1+2*k2+2*k3+k4)/6
		self.y2 = self.y1 + dy

	### evolution x
		part1 = self.vx_B1 * ( np.cosh(self.b1*(self.y1-self.w_particle)) + np.cosh(self.b1*(self.y2-self.w_particle)) )
		part3 = self.vx_B3 * ( np.cosh(self.b3*(self.y1-self.w_particle)) + np.cosh(self.b3*(self.y2-self.w_particle)) )
		part5 = self.vx_B5 * ( np.cosh(self.b5*(self.y1-self.w_particle)) + np.cosh(self.b5*(self.y2-self.w_particle)) )

		vx = self.vx_U * (self.vx_A1 * (2*self.size_particle - part1) +
						  self.vx_A3 * (2*self.size_particle - part3) +
						  self.vx_A5 * (2*self.size_particle - part5) )
		self.x = self.x + vx*dt

	### reorganistion
		self.y1 = np.copy(self.y2)
		self.time = self.time + dt
		self.counter += int(1)




### comportment at intersection ###
	def update( self, geo, vector ):
	### update value after an intersection
		self.lvl_particle_branch[vector] += 1
		self.w_particle[vector] = geo.w_branch[self.lvl_particle_branch[vector]]
		self.v_particle[vector] = geo.v_branch[self.lvl_particle_branch[vector]]

		trueTop = vector * self.particle_top
		if np.any(trueTop):
			if geo.lvl_max == 1 :
				self.nbr_particle_branch[trueTop] = geo.connectivity[0][1]
			else :
				f = np.where( trueTop == True )
				for ii in f[0] :
					self.nbr_particle_branch[ii] = geo.connectivity[self.nbr_particle_branch[ii]-1][1]
			self.y1[trueTop] = 2*self.w_particle[trueTop] - 2*self.y1[trueTop]

		trueBot = vector * np.invert(self.particle_top)
		if np.any(trueBot):
			if geo.lvl_max == 1 :
				self.nbr_particle_branch[trueBot] = geo.connectivity[0][2]
			else :
				f = np.where( trueBot == True )
				for ii in f[0] :
					self.nbr_particle_branch[ii] = geo.connectivity[self.nbr_particle_branch[ii]-1][2]
			self.y1[trueBot] = 2*self.y1[trueBot]

		self.y1[self.y1 < self.lim_wall] = self.lim_wall
		self.y1[self.y1 > 2*self.w_particle-self.lim_wall] = 2*self.w_particle[self.y1>2*self.w_particle-self.lim_wall] - self.lim_wall

		self.particle_top[vector] = False
		self.particle_top[self.y1>self.w_particle] = True
		self.y1[vector*self.particle_top] = 2*self.w_particle[vector*self.particle_top] - self.y1[vector*self.particle_top]

		### constant for integration
		self.v0[vector] = self.v_particle[vector] / (1 * (1-1 / np.cosh( self.b1 * self.w_particle[vector] ))-
									 1 / 9 * (1-1 / np.cosh( self.b3 * self.w_particle[vector] ))+
									 1 / 25 * (1-1 / np.cosh( self.b5 * self.w_particle[vector] )))

		# vy = mu_U/y**delta * sum(mu_An * sinh(bn*(y-w))
		self.mu_U[vector] = self.xi * self.size_particle**(self.delta+1) * self.v0[vector]
		self.mu_A1[vector] = -1 * self.b1 / np.cosh( self.b1 * self.w_particle[vector] )
		self.mu_A3[vector] = 1 / 9 * self.b3 / np.cosh( self.b3 * self.w_particle[vector] )
		self.mu_A5[vector] = -1 / 25 * self.b5 / np.cosh( self.b5 * self.w_particle[vector] )

		# vx = vx_U * sum(vx_An * (2*r - vx_Bn*(cosh(bn*(y1-w))+cosh(bc*(y2-w)) ) )
		self.vx_U[vector] = self.v0[vector] * geo.height / (2 * pi * self.size_particle**2)
		self.vx_B1[vector] = np.sinh(self.b1*self.size_particle) / (self.b1*np.cosh( self.b1 * self.w_particle[vector] ))
		self.vx_B3[vector] = np.sinh(self.b3*self.size_particle) / (self.b3*np.cosh( self.b3 * self.w_particle[vector] ))
		self.vx_B5[vector] = np.sinh(self.b5*self.size_particle) / (self.b5*np.cosh( self.b5 * self.w_particle[vector] ))



### comportment at intersection ###
	def final( self ):
		self.y1[self.particle_top] = 2*self.w_particle[self.particle_top] - self.y1[self.particle_top]


### time at passing the final line ###
	def time_to_finish(self, test) :
		self.time_finish[test] = self.time
		self.finish_or_not[test] = False


### save data ###
	def savedata( self ):
		self.save_x = np.append( self.save_x, self.x.reshape((1,-1)), axis=0 )
		self.save_y = np.append( self.save_y, self.y1.reshape((1,-1)), axis=0 )
		#self.save_y[self.particle_top, -1] = 2*self.w_particle[self.particle_top] - self.y1[self.particle_top]
		self.save_lvl = np.append( self.save_lvl, self.lvl_particle_branch.reshape((1,-1)), axis=0 )
		self.save_nbr_channel = np.append( self.save_nbr_channel, self.nbr_particle_branch.reshape((1,-1)), axis=0 )










