import time
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
matplotlib.use("Agg")
import sys
import itertools
import h5py
import json
import cPickle as pickle

from pompy import models
import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt

file_name = 'near_plume_sutton'
output_file = file_name+'.pkl'

dt = 0.25
frame_rate = 8
times_real_time = 2 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))
simulation_time = 95. #seconds
release_delay=0.

#traps
trap_param = {
        'source_locations' : [(7.5,25.),],
        'source_strengths' : [1.,],
        'epsilon'          : 0.01,
        'trap_radius'      : 0.5,
        'source_radius'    : 0.
}

traps = trap_models.TrapModel(trap_param)

source_pos = np.array([np.array(tup) for tup in trap_param['source_locations']])


Q,C_y,n = (5.,0.4,0.8)
wind_angle = -2*np.pi/3
wind_mag = 1.

suttonPlumes = models.SuttonModelPlume(Q,C_y,n,source_pos,wind_angle)

wind_param = {
            'speed': wind_mag,
            'angle': wind_angle,
            'evolving': False,
            'wind_dt': None,
            'dt': dt
            }
wind_field = wind_models.WindField(param=wind_param)


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
            'upper': 0.8,  # detection probability/sec of exposure
            },
        'schmitt_trigger':True,
        'low_pass_filter_length':3, #seconds
        'dt_plot': capture_interval*dt,
        't_stop':simulation_time
        }
swarm = swarm_models.BasicSwarmOfFlies(wind_field,traps,param=swarm_param,
    start_type='fh',track_plume_bouts=False,track_arena_exits=False)

#Plotting concentration
xlim = (-15., 15.)
ylim = (0., 40.)
sim_region_tuple = xlim[0], xlim[1], ylim[0], ylim[1]
sim_region = models.Rectangle(*sim_region_tuple)

im_extents = sim_region_tuple
xmin,xmax,ymin,ymax = im_extents
# Set up figure
plt.ion()
fig = plt.figure(figsize=(7.5, 9))
ax = fig.add_subplot(111)


#Initial concentration plotting
conc_samples = suttonPlumes.conc_im(im_extents)

cmap = matplotlib.colors.ListedColormap(['white', 'orange'])

conc_im = plt.imshow(conc_samples,cmap = cmap,extent=im_extents,origin='lower')

log_im = scipy.log(conc_samples)
cutoff_l = scipy.percentile(log_im[~scipy.isinf(log_im)],70)
cutoff_u = scipy.percentile(log_im[~scipy.isinf(log_im)],99)

conc_im.set_data(log_im)
n = matplotlib.colors.Normalize(vmin=cutoff_l,vmax=cutoff_u)
conc_im.set_norm(n)

buffr = 4
ax.set_xlim((xmin-buffr,xmax+buffr))
ax.set_ylim((ymin-buffr,ymax+buffr))

#Initial fly plotting
#Sub-dictionary for color codes for the fly modes
Mode_StartMode = 0
Mode_FlyUpWind = 1
Mode_CastForOdor = 2
Mode_Trapped = 3

edgecolor_dict = {Mode_StartMode : 'blue',
Mode_FlyUpWind : 'red',
Mode_CastForOdor : 'red',
Mode_Trapped :   'black'}

facecolor_dict = {Mode_StartMode : 'blue',
Mode_FlyUpWind : 'red',
Mode_CastForOdor : 'white',
Mode_Trapped :   'black'}


fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]
fly_dots = plt.scatter(swarm.x_position, swarm.y_position,
    edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)


#Put the time in the corner
(xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
text = '0 min 0 sec'
timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')


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
        swarm.update(t,dt,wind_field,suttonPlumes,traps,pre_stored=True) #for presaved plumes
        #Update time display
        text ='{0} min {1} sec'.format(int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
        timer.set_text(text)
        t+= dt
        time.sleep(0.001)
    # Update live display

    fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

    fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
    fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]

    fly_dots.set_edgecolor(fly_edgecolors)
    fly_dots.set_facecolor(fly_facecolors)

    trap_list = []
    for trap_num, trap_loc in enumerate(traps.param['source_locations']):
        mask_trap = swarm.trap_num == trap_num
        trap_cnt = mask_trap.sum()
        trap_list.append(trap_cnt)
    total_cnt = sum(trap_list)
    title.set_text('{0}/{1}'.format(total_cnt,swarm.size))

    writer.grab_frame()
    fig.canvas.flush_events()


writer.finish()

with open(output_file, 'w') as f:
    pickle.dump(swarm,f)


    # anim = FuncAnimation(fig, update, frames=frames, repeat=False,save_count = frames)
    # plt.show()
    # saved = anim.save(file_name+'.mp4', dpi=100, fps=frame_rate,
    # extra_args=['-vcodec', 'libx264'])
