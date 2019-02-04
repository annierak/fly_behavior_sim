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
import scipy.sparse

dgfile = 'fly_density_timecourse_10.23-22:19_10_26_fly_sim.hdf5'

density_grid = density_grid_loaders.DensityLoader(dgfile)

times = np.arange(0,density_grid.t_stop,density_grid.dt_store)

grid_x_size,grid_y_size = np.shape(density_grid.value(0))
density_grid_array = np.zeros((grid_x_size,grid_y_size,len(times)))

for i,t in enumerate(times):
    density_grid_array[:,:,i] = density_grid.value(t)

print('Size of regular array is %s MB' %  (density_grid_array.nbytes/1e6))

density_grid_array = np.reshape(density_grid_array,(grid_x_size*grid_y_size,len(times)))

plt.spy(density_grid_array)
plt.show()

sparse_array = scipy.sparse.csr_matrix(density_grid_array)

print('Size of sparse array is %s MB' %  ((
    sparse_array.data.nbytes +
    sparse_array.indptr.nbytes + sparse_array.indices.nbytes)/1e6))
