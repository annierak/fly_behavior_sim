import time
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
matplotlib.use("Agg")
import sys
import itertools
import json
import cPickle as pickle
from matplotlib.animation import FuncAnimation
import datetime

import h5_logger
import h5py
import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
#This is hacky -- should turn data_importers into a module
sys.path.insert(0, '../')
import data_importers
import density_grid_loaders

dgfile = 'fly_density_timecourse_10.23-22:19_10_26_fly_sim.hdf5'

density_grid = density_grid_loaders.DensityLoader(dgfile)

fig=plt.figure(10)
imd = plt.imshow(np.random.randn(1000,1000))
timer = plt.text(0.5,0.85,'0 s',transform=fig.transFigure)
plt.ion()

for t in np.arange(0,density_grid.t_stop,density_grid.dt_store):
    grid_at_t = density_grid.value(t)
    print(np.unique(grid_at_t))
    print(np.sum(grid_at_t>0.))
    time.sleep(.1)
    imd.set_data(grid_at_t)
    text ='{0} min {1} sec'.format(int(scipy.floor(
            t/60.)),int(scipy.floor(t%60.)))
    timer.set_text(text)
    plt.pause(.001)
