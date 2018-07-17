#This runs a simulation and plots a comparison of the odor values computed using
#the original pompy direct computation and the odor values computed using the
#box method approximation.

import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
import h5py
import json
import cPickle as pickle

import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
import data_importers

dt = 1.
simulation_time = 30*60. #seconds
release_delay = 20.*60

#Import wind and odor fields
conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
wind_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]
plume_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[3]

importedConc = data_importers.ImportedConc(conc_file,release_delay)
importedWind = data_importers.ImportedWind(wind_file,release_delay)

xmin,xmax,ymin,ymax = importedConc.simulation_region


array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.

importedPlumesExact = data_importers.ImportedPlumes(plume_file,
    array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay)

importedPlumesApprox = data_importers.ImportedPlumes(plume_file,
    array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay,box_approx=True)


target_size = 1000
odor_comp_collector = np.zeros((2,int(target_size*simulation_time/dt)))
counter = 0

t = 0

while t<simulation_time:
    t0 = time.time()
    puff_array = importedPlumesExact.puff_array_at_time(t)
    time.sleep(0.01)
    t+=dt
    x_position = np.random.uniform(xmin,xmax,target_size)
    y_position = np.random.uniform(ymin,ymax,target_size)
    odor_direct = importedPlumesExact.value(t,x_position,y_position)
    odor_approx = importedPlumesApprox.value(t,x_position,y_position)
    odor_comp_collector[:,counter*target_size:\
        counter*target_size+target_size] = odor_direct,odor_approx
    counter +=1
    print(t,time.time()-t0)

plt.figure(1)
plt.scatter(odor_comp_collector[0,:],odor_comp_collector[1,:])
plt.show()
