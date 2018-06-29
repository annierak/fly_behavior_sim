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

file_name = 'faster_dt_test'
output_file = file_name+'.pkl'

#traps
trap_param = {
        'source_locations' : [(7.5,25.),],
        'source_strengths' : [1.,],
        'epsilon'          : 0.01,
        'trap_radius'      : 0.5,
        'source_radius'    : 0.
}

traps = trap_models.TrapModel(trap_param)

#Import wind and odor fields
conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
wind_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]
plume_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[3]
release_delay = 10.

importedConc = data_importers.ImportedConc(conc_file,release_delay)
importedWind = data_importers.ImportedWind(wind_file,release_delay)

array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.

importedPlumes = data_importers.ImportedPlumes(plume_file,
    array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay)


dt = 0.25
frame_rate = 8
times_real_time = 2 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))
print(capture_interval)
simulation_time = 95. #seconds

#Setup fly swarm
release_time_constant=0.1
wind_slippage = (0.,0.)
swarm_size=100

swarm_param = {
        'swarm_size'          : swarm_size,
        'heading_data'        : None,
        'initial_heading_dist': scipy.radians(90.),
        'initial_heading'     : scipy.random.uniform(scipy.radians(80.),scipy.radians(100.),swarm_size),
        'x_start_position'    : scipy.random.uniform(2,5,swarm_size),
        'y_start_position'    : 13.*scipy.ones((swarm_size,)),
        'heading_error_std'   : scipy.radians(10.0),
        'flight_speed'        : scipy.full((swarm_size,), 0.5),
        'release_time'        : scipy.random.exponential(release_time_constant,(swarm_size,)),
        'release_time_constant': release_time_constant,
        'release_delay'       : release_delay,
        'cast_interval'       : [1, 3],
        'wind_slippage'       : wind_slippage,
        'odor_thresholds'     : {
            'lower': 0.0005,
            'upper': 1    
            },
        'odor_probabilities'  : {
            'lower': 0.9,    # detection probability/sec of exposure
            'upper': 0.002,  # detection probability/sec of exposure
            },
        'schmitt_trigger':False,
        'low_pass_filter_length':3, #seconds
        'dt_plot': capture_interval*dt,
        't_stop':simulation_time
        }
swarm = swarm_models.BasicSwarmOfFlies(importedWind,traps,param=swarm_param,
    start_type='fh',track_plume_bouts=False,track_arena_exits=False)

vmin,vmax,cmap = importedConc.get_image_params()
im_extents = importedConc.simulation_region
xmin,xmax,ymin,ymax = im_extents
# Set up figure
plt.ion()
fig = plt.figure(figsize=(7.5, 9))
ax = fig.add_subplot(111)

#Initial concentration plotting
image = importedConc.plot(0)
buffr = 4
ax.set_xlim((xmin-buffr,xmax+buffr))
ax.set_ylim((ymin-buffr,ymax+buffr))

#Initial fly plotting
#Sub-dictionary for color codes for the fly modes
Mode_StartMode = 0
Mode_FlyUpWind = 1
Mode_CastForOdor = 2
Mode_Trapped = 3

color_dict = {Mode_StartMode : 'blue',
Mode_FlyUpWind : 'red',
Mode_CastForOdor : 'orange',
Mode_Trapped :   'black'}


fly_colors = [color_dict[mode] for mode in swarm.mode]
fly_dots = plt.scatter(swarm.x_position, swarm.y_position,color=fly_colors,alpha=0.5)

#Put the time in the corner
(xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
text = '0 min 0 sec'
timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')

#Initial wind plotting
x_coords,y_coords = importedWind.get_plotting_points()
u,v = importedWind.quiver_at_time(0)
vector_field = ax.quiver(x_coords,y_coords,u,v)

#title with trapped count
trap_list = []
for trap_num, trap_loc in enumerate(traps.param['source_locations']):
    mask_trap = swarm.trap_num == trap_num
    trap_cnt = mask_trap.sum()
    trap_list.append(trap_cnt)
total_cnt = sum(trap_list)
title = plt.title('{0}/{1}'.format(total_cnt,swarm.size))

# frames = int(frame_rate*simulation_time/times_real_time)+1

FFMpegWriter = animate.writers['ffmpeg']
metadata = {'title':file_name,}
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, file_name+'.mp4', 500)


t = 0.0 #- release_delay

while t<simulation_time:
    for k in range(capture_interval):
        #update flies
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        # swarm.update(t,dt,importedWind,importedConc,traps,pre_stored=True) #for presaved conc
        swarm.update(t,dt,importedWind,importedPlumes,traps,pre_stored=True) #for presaved plumes
        #Update time display
        text ='{0} min {1} sec'.format(int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
        timer.set_text(text)
        t+= dt
        time.sleep(0.001)
    # Update live display
    '''plot the flies'''
    #plot the wind vector field
    u,v = importedWind.quiver_at_time(t)
    vector_field.set_UVC(u,v)

    fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

    fly_colors = [color_dict[mode] for mode in swarm.mode]
    fly_dots.set_color(fly_colors)

    trap_list = []
    for trap_num, trap_loc in enumerate(traps.param['source_locations']):
        mask_trap = swarm.trap_num == trap_num
        trap_cnt = mask_trap.sum()
        trap_list.append(trap_cnt)
    total_cnt = sum(trap_list)
    title.set_text('{0}/{1}'.format(total_cnt,swarm.size))

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
