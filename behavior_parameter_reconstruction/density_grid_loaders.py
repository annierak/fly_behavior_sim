
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


class DensityLoader(object):
    def __init__(self,hdf5_file):
        self.data = h5py.File(hdf5_file,'r')
        run_param = json.loads(self.data.attrs['jsonparam'])
        for key,value in run_param.items():
            if isinstance(value,list):
                run_param[key] = np.array(value)
        self.dt_store = run_param['sim_dt']
        self.t_stop = run_param['sim_duration']
        self.grid_hist = self.data['grid_hist']
    def value(self,t):
        ind = int(scipy.floor((t)/self.dt_store))
        return self.grid_hist[ind,:,:]
