import time
import scipy
import matplotlib
matplotlib.use("Agg") #This needs to be placed before importing any sub-packages
#of matplotlib or else the double animate problem happens
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import sys


import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.simulation_running_tools as srt
import data_importers

plume_file = '/home/annie/work/programming/pompy_duplicate/puffObject6.5-22:6.hdf5'

array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.
release_delay = 20.*60


importedPlumes = data_importers.ImportedPlumes(plume_file,
    array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay)

dt = 10
t = 0.
t_stop = 60*50.

while t<t_stop:
    puff_array = importedPlumes.puff_array_at_time(t)
    radii = puff_array[:,3]
    print(t/60.)
    print(len(radii))
    biggest = max(radii)
    print(biggest)
    time.sleep(0.1)
    t+=dt
