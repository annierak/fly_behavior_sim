
import numpy as np

import h5_logger
import h5py
import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt

keys = ['a'+str(i) for i in range(10)]
vals = [np.random.randn(3,3) for i in range(10)]

test_dict= dict(zip(keys,vals))

print(test_dict)

logger = h5_logger.H5Logger('test_log.hdf5',param_attr=test_dict)

data = {'a':[1,2,3]}

logger.add(data)
