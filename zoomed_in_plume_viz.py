import time
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
matplotlib.use("Agg")
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

file_name = 'hdf5_plume_test'
output_file = file_name+'.pkl'

number_sources = 8
radius_sources = 1000.0
trap_radius = 0.5
location_list, strength_list = utility.create_circle_of_sources(number_sources,
                radius_sources,None)
trap_param = {
        'source_locations' : location_list,
        'source_strengths' : strength_list,
        'epsilon'          : 0.01,
        'trap_radius'      : trap_radius,
        'source_radius'    : radius_sources
}

traps = trap_models.TrapModel(trap_param)

#Import wind and odor fields
conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
wind_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]
plume_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[3]

importedConc = data_importers.ImportedConc(conc_file)
importedWind = data_importers.ImportedWind(wind_file)

array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.

importedPlumes = data_importers.ImportedPlumes(plume_file,
    array_z,array_dim_x,array_dim_y,puff_mol_amount)


dt = 0.25
frame_rate = 8
times_real_time = 2 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))
print(capture_interval)
simulation_time = 60. #seconds


vmin,vmax,cmap = importedConc.get_image_params()
im_extents = importedConc.simulation_region
xmin,xmax,ymin,ymax = 500,1200,0,500
# Set up figure
plt.ion()
fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111)

#Initial concentration plotting
image = importedConc.plot(0)
buffr = -10
ax.set_xlim((xmin-buffr,xmax+buffr))
ax.set_ylim((ymin-buffr,ymax+buffr))


#Put the time in the corner
(xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
text = '0 min 0 sec'
timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')

# #Initial wind plotting
# x_coords,y_coords = importedWind.get_plotting_points()
# u,v = importedWind.quiver_at_time(0)
# vector_field = ax.quiver(x_coords,y_coords,u,v)



# frames = int(frame_rate*simulation_time/times_real_time)+1

FFMpegWriter = animate.writers['ffmpeg']
metadata = {'title':file_name,}
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, file_name+'.mp4', 500)


t = 0.0

while t<simulation_time:
    for k in range(capture_interval):
        #update flies
        print('t: {0:1.2f}'.format(t))
        #Update time display
        text ='{0} min {1} sec'.format(int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
        timer.set_text(text)
        t+= dt
        time.sleep(0.001)
    # Update live display
    '''plot the flies'''
    #plot the wind vector field
    # u,v = importedWind.quiver_at_time(t)
    # vector_field.set_UVC(u,v)



    '''plot the odor concentration field'''
    conc_array = importedConc.array_at_time(t)
    image.set_data(conc_array)
    plt.pause(0.001)

    writer.grab_frame()
    fig.canvas.flush_events()


writer.finish()

with open(output_file, 'w') as f:
    pickle.dump(swarm,f)


    # anim = FuncAnimation(fig, update, frames=frames, repeat=False,save_count = frames)
    # plt.show()
    # saved = anim.save(file_name+'.mp4', dpi=100, fps=frame_rate,
    # extra_args=['-vcodec', 'libx264'])
