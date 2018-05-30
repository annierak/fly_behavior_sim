import time
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import sys
import itertools
import h5py
import json
import cPickle as pickle


import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
import data_importers

file_name = 'imported_plumes_test'

fig = plt.figure(1)
fig.set_size_inches(7.5,9,True)
ax = fig.add_subplot(111)
plt.pause(0.001)
t = 0.0

conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
importedConc = data_importers.ImportedConc(conc_file)
image = importedConc.plot(0)
xmin,xmax,ymin,ymax=image.get_extent()
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

# wind_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]
# importedWind = data_importers.ImportedWind(wind_file)
# x_coords,y_coords = importedWind.get_plotting_points()
# u,v = importedWind.quiver_at_time(0)
# vector_field = ax.quiver(x_coords,y_coords,u,v)

dt = 0.25
frame_rate = 20
times_real_time = 5 # seconds of simulation / sec in video
capture_interval = times_real_time*int((1./frame_rate)/dt)
simulation_time = 60*2. #seconds

def update(i):
    print(i)
    global t

    #plot the wind vector field
    # u,v = importedWind.quiver_at_time(t)
    # vector_field.set_UVC(u,v)

    '''plot the odor concentration field'''
    conc_array = importedConc.array_at_time(t)
    image.set_data(conc_array)

    t+=dt
    return [image]

anim = FuncAnimation(fig, update, frames=int(frame_rate*simulation_time/times_real_time), repeat=False)
saved = anim.save(file_name+'.mp4', dpi=100, fps=frame_rate, extra_args=['-vcodec', 'libx264'])
