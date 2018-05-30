import h5py
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
import data_importers
from odor_tracking_sim import fly_models
import scipy
import time

conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
wind_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]

importedConc = data_importers.ImportedConc(conc_file)
importedWind = data_importers.ImportedWind(wind_file)


vmin,vmax,cmap = importedConc.get_image_params()
im_extents = importedConc.simulation_region
example_conc_frame = importedConc.array_at_time(10)

for k in range(100):
    print(importedConc.value(k*0.25,scipy.array([5]),scipy.array([20])))
    time.sleep(1)

plt.figure(1)
plt.imshow(example_conc_frame,vmin=vmin,vmax=vmax,cmap=cmap,extent=im_extents)
print(im_extents)
x_coords,y_coords = importedWind.get_plotting_points()
u,v = importedWind.quiver_at_time(10)
print(np.shape(u))
plt.quiver(x_coords,y_coords,u,v)
plt.show()

plt.figure(2)
ax = plt.subplot(1,1,1)
image = importedConc.plot(0)
im_extents = importedConc.simulation_region
xmin,xmax,ymin,ymax = im_extents
buffr = 4
ax.set_xlim((xmin-buffr,xmax+buffr))
ax.set_ylim((ymin-buffr,ymax+buffr))

plt.show()
