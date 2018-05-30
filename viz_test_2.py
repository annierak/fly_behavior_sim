from pompy import models, processors
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.animation import FuncAnimation
import scipy
import utility
import sys
import time
import itertools
import cPickle as pickle
import h5py
import json
import cPickle as pickle


import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
import data_importers


dt = 0.25
frame_rate = 20
times_real_time = 5 # seconds of simulation / sec in video
capture_interval = times_real_time*int((1./frame_rate)/dt)
simulation_time = 60*2. #seconds


#Odor arena
xlim = (-15., 15.)
ylim = (0., 40.)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
wind_region = models.Rectangle(xlim[0]*1.2,ylim[0]*1.2,
xlim[1]*1.2,ylim[1]*1.2)
source_pos = (7.5,25)





# Set up figure
fig = plt.figure(figsize=(7.5, 9))
ax = fig.add_axes([0., 0., 1., 1.])
buffr = 4
ax.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
ax.set_ylim((ylim[0]-buffr,ylim[1]+buffr))

#Concentration extraction
conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
importedConc = data_importers.ImportedConc(conc_file)
conc_im = importedConc.plot(0)
xmin,xmax,ymin,ymax=conc_im.get_extent()

# Define animation update function
t=0
def update(i):
    print(i)
    global t
    for m in range(10):
        print(m)
        t+=1

    conc_array = importedConc.array_at_time(t)
    conc_im.set_data(conc_array)

    return [conc_im]#,vector_field]

# Run and save output to video
anim = FuncAnimation(fig, update, frames=int(frame_rate*simulation_time/times_real_time), repeat=False)

plt.show()

#Save the animation to video
saved = anim.save('plume_saving_test.mp4', dpi=100, fps=frame_rate, extra_args=['-vcodec', 'libx264'])
