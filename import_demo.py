import h5py
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np

conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
conc_data = h5py.File(conc_file,'r')
big_conc_array = conc_data['conc_array']

example_frame = big_conc_array[100,:,:]
plt.figure(1)
plt.imshow(example_frame)
plt.show()

wind_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]
wind_data = h5py.File(wind_file,'r')
big_wind_array = wind_data['velocity_field']
print(np.shape(big_wind_array))
