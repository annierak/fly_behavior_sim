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


import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
from pompy import models,processors

file_name = 'coarse_dt_live_near_plume'
output_file = file_name+'.pkl'

dt = 0.25
plume_dt = 0.25
frame_rate = 8
times_real_time = 2 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))
simulation_time = 95. #seconds
release_delay = 30.

#traps
trap_param = {
        'source_locations' : [(7.5,25.),],
        'source_strengths' : [1.,],
        'epsilon'          : 0.01,
        'trap_radius'      : 0.5,
        'source_radius'    : 0.
}

traps = trap_models.TrapModel(trap_param)

#Region
xlim = (-15., 15.)
ylim = (0., 40.)

sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
source_pos = scipy.array([scipy.array(tup) for tup in traps.param['source_locations']]).T


#wind model setup
constant_wind_angle = -2*np.pi/3
wind_mag = 1.

wind_region = models.Rectangle(xlim[0]*1.2,ylim[0]*1.2,
    xlim[1]*1.2,ylim[1]*1.2)
diff_eq = False
aspect_ratio= (xlim[1]-xlim[0])/(ylim[1]-ylim[0])
noise_gain=3.
noise_damp=0.071
noise_bandwidth=0.71
wind_grid_density = 200
Kx = Ky = 10000 #highest value observed to not cause explosion: 10000
wind_field = models.WindModel(wind_region,int(wind_grid_density*aspect_ratio),
wind_grid_density,noise_gain=noise_gain,noise_damp=noise_damp,
noise_bandwidth=noise_bandwidth,Kx=Kx,Ky=Ky,
diff_eq=diff_eq,angle=constant_wind_angle,mag=wind_mag)

#Wind model for flies
wind_param = {
            'speed': wind_mag,
            'angle': constant_wind_angle,
            'evolving': False,
            'wind_dt': None,
            'dt': dt
            }
wind_field_flies = wind_models.WindField(param=wind_param)

# Set up plume model
centre_rel_diff_scale = 2.
# puff_release_rate = 0.001
puff_release_rate = 10
puff_spread_rate=0.005
puff_init_rad = 0.01
max_num_puffs=int(2e5)
# max_num_puffs=100

plume_model = models.PlumeModel(
    sim_region, source_pos, wind_field,simulation_time+release_delay,
    centre_rel_diff_scale=centre_rel_diff_scale,
    puff_release_rate=puff_release_rate,
    puff_init_rad=puff_init_rad,puff_spread_rate=puff_spread_rate,
    max_num_puffs=max_num_puffs)

# Create a concentration array generator
array_z = 0.01

array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.
array_gen = processors.ConcentrationArrayGenerator(
    sim_region, array_z, array_dim_x, array_dim_y, puff_mol_amount)


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
swarm = swarm_models.BasicSwarmOfFlies(wind_field_flies,traps,param=swarm_param,
    start_type='fh',track_plume_bouts=False,track_arena_exits=False)

sim_region_tuple = xlim[0], xlim[1], ylim[0], ylim[1]
im_extents = sim_region_tuple
# Set up figure
plt.ion()
fig = plt.figure(figsize=(7.5, 9))
ax = fig.add_subplot(111)

#Initial concentration plotting
conc_array = array_gen.generate_single_array(plume_model.puffs)
xmin = sim_region.x_min; xmax = sim_region.x_max
ymin = sim_region.y_min; ymax = sim_region.y_max
# im_extents = (xmin,xmax,ymin,ymax)
vmin,vmax = 0.,50.
cmap = matplotlib.colors.ListedColormap(['white', 'orange'])
cmap = 'Reds'
conc_im = ax.imshow(conc_array.T[::-1], extent=im_extents,
vmin=vmin, vmax=vmax, cmap=cmap)

buffr = 4
ax.set_xlim((xmin-buffr,xmax+buffr))
ax.set_ylim((ymin-buffr,ymax+buffr))

#Conc array gen to be used for the flies
sim_region_tuple = plume_model.sim_region.as_tuple()
box_min,box_max = sim_region_tuple[1],sim_region_tuple[2]

r_sq_max=20;epsilon=0.00001;N=1e6

array_gen_flies = processors.ConcentrationValueCalculator(
            puff_mol_amount)

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


t = 0.0 - release_delay

while t<simulation_time:
    for k in range(capture_interval):
        #update flies
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        for j in range(int(dt/plume_dt)):
            wind_field.update(plume_dt)
            plume_model.update(plume_dt,verbose=True)
        if t>0.:
            swarm.update(t,dt,wind_field_flies,array_gen_flies,traps,plumes=plume_model,
                pre_stored=False)
        t+= dt
    # Update live display
    '''plot the flies'''

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
    conc_array = array_gen.generate_single_array(plume_model.puffs)
    conc_im.set_data(conc_array[::-1])
    plt.pause(0.001)

    # writer.grab_frame()


writer.finish()

with open(output_file, 'w') as f:
    pickle.dump(swarm,f)


    # anim = FuncAnimation(fig, update, frames=frames, repeat=False,save_count = frames)
    # plt.show()
    # saved = anim.save(file_name+'.mp4', dpi=100, fps=frame_rate,
    # extra_args=['-vcodec', 'libx264'])
