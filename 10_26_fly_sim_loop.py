#This is a looped copy of 10_26_fly_sim_FuncAnimation.py
#It does not record video

import time
import scipy
import matplotlib
matplotlib.use("Agg") #This needs to be placed before importing any sub-packages
#of matplotlib or else the double animate problem happens
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import sys
import itertools
import h5py
import json
import cPickle as pickle
from matplotlib.animation import FuncAnimation


import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
import data_importers


descriptor = '_cast_delay_05'
repeat_count = 10

file_name = '10_26_fly_sim'+descriptor

dt = 0.25
frame_rate = 20
times_real_time = 30 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))


simulation_time = 50*60. #seconds
release_delay = 20.*60

#traps
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

importedConc = data_importers.ImportedConc(conc_file,release_delay)
importedWind = data_importers.ImportedWind(wind_file,release_delay)

array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.

importedPlumes = data_importers.ImportedPlumes(plume_file,
    array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay)



#Setup fly swarm
wind_slippage = (1.0,1.0)
swarm_size=2000

#construct release distribution
empirical_release_data_file = 'empirical_release_counts.pkl'
with open(empirical_release_data_file,'r') as f:
    release_data_counts = pickle.load(f)

#turn release counts into a list of times
flattened_release_counts = scipy.sum(release_data_counts,0).astype(int)
emp_dt = 1./25
times = scipy.linspace(dt,len(flattened_release_counts)*emp_dt,
    len(flattened_release_counts))
total_count = int(sum(flattened_release_counts))
release_times = scipy.zeros(total_count)
counter = 0
for release_time,count in list(zip(times,flattened_release_counts)):
    if count>0:
        release_times[counter:counter+count] = release_time
        counter +=count

release_times = utility.draw_from_inputted_distribution(
    release_times,2,swarm_size)


heading_data = {'angles':(scipy.pi/180)*scipy.array([0.,90.,180.,270.]),
                'counts':scipy.array([[1724,514,1905,4666],[55,72,194,192]])
                }

cast_timeout = 20
cast_interval = [1,3]
cast_delay= 0.5
flight_speed = 1.5
odor_threshold = 0.05


swarm_param = {
        'swarm_size'          : swarm_size,
        'heading_data'        : heading_data,
        'x_start_position'    : scipy.zeros(swarm_size),
        'y_start_position'    : scipy.zeros(swarm_size),
        'flight_speed'        : scipy.full((swarm_size,), flight_speed),
        'release_time'        : release_times,
        'release_delay'       : 0.,
        'cast_interval'       : cast_interval,
        'wind_slippage'       : wind_slippage,
        'odor_thresholds'     : {
            'lower': 0.0005,
            'upper': odor_threshold
            },
        'schmitt_trigger':False,
        'low_pass_filter_length':cast_delay, #seconds
        'dt_plot': capture_interval*dt,
        't_stop':simulation_time,
        'cast_timeout':cast_timeout
        }
swarm = swarm_models.BasicSwarmOfFlies(importedWind,traps,param=swarm_param,
    start_type='fh',track_plume_bouts=False,track_arena_exits=False)


# vmin,vmax,cmap = importedConc.get_image_params()
# im_extents = importedConc.simulation_region
# xmin,xmax,ymin,ymax = im_extents
# # Set up figure
# # plt.ion()
fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111)
#
# #Initial concentration plotting
# image = importedConc.plot(0,vmin=0.,vmax=1.)
# buffr = -300
# ax.set_xlim((xmin-buffr,xmax+buffr))
# ax.set_ylim((ymin-buffr,ymax+buffr))
#
# #Initial fly plotting
# #Sub-dictionary for color codes for the fly modes
# Mode_StartMode = 0
# Mode_FlyUpWind = 1
# Mode_CastForOdor = 2
# Mode_Trapped = 3
#
# color_dict = {Mode_StartMode : 'blue',
# Mode_FlyUpWind : 'red',
# Mode_CastForOdor : 'orange',
# Mode_Trapped :   'black'}
#
#
# fly_colors = [color_dict[mode] for mode in swarm.mode]
# fly_dots = plt.scatter(swarm.x_position, swarm.y_position,color=fly_colors,alpha=0.5)
#
# #Put the time in the corner
# (xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
# text = '0 min 0 sec'
# timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')
#
# #Initial wind plotting -- subsampled
# u,v = importedWind.quiver_at_time(0)
# full_size = scipy.shape(u)[0]
# print(full_size)
# shrink_factor = 10
# u,v = u[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor],\
#     v[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor]
# x_origins,y_origins = importedWind.get_plotting_points()
# x_origins,y_origins = x_origins[0:-1:full_size-1],\
#     y_origins[0:-1:full_size-1]
# x_origins,y_origins = x_origins[0:full_size-1:shrink_factor],\
#     y_origins[0:full_size-1:shrink_factor]
# coords = scipy.array(list(itertools.product(x_origins, y_origins)))
# x_coords,y_coords = coords[:,0],coords[:,1]

# vector_field = ax.quiver(x_coords,y_coords,u,v)
# #title with trapped count
# trap_list = []
# for trap_num, trap_loc in enumerate(traps.param['source_locations']):
#     mask_trap = swarm.trap_num == trap_num
#     trap_cnt = mask_trap.sum()
#     trap_list.append(trap_cnt)
# total_cnt = sum(trap_list)
# title = plt.title('{0}/{1}:{2}'.format(total_cnt,swarm.size,trap_list))

# frames = int(frame_rate*simulation_time/times_real_time)+1

def init():
    #do nothing
    pass


def update(i):
    print('i='+str(i))
    global swarm, t, release_delay,importedConc,importedWind,importedPlumes,\
        traps#,fly_dots,title,image
    for k in range(capture_interval):
        #update flies
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        # swarm.update(t,dt,importedWind,importedConc,traps,pre_stored=True) #for presaved conc

        swarm.update(t,dt,importedWind,importedPlumes,traps,pre_stored=True) #for presaved plumes
        # text ='{0} min {1} sec'.format(int(scipy.floor(
        #         t/60.)),int(scipy.floor(t%60.)))
        # timer.set_text(text)
        t+= dt
        time.sleep(0.001)
    # Update live display
    '''plot the flies'''
    # #plot the wind vector field
    # u,v = importedWind.quiver_at_time(t)
    # u,v = u[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor],\
    #     v[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor]
    # vector_field.set_UVC(u,v)
    #
    # fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])
    #
    # fly_colors = [color_dict[mode] for mode in swarm.mode]
    # fly_dots.set_color(fly_colors)
    #
    # trap_list = []
    # for trap_num, trap_loc in enumerate(traps.param['source_locations']):
    #     mask_trap = swarm.trap_num == trap_num
    #     trap_cnt = mask_trap.sum()
    #     trap_list.append(trap_cnt)
    # total_cnt = sum(trap_list)
    # title.set_text('{0}/{1}:{2}'.format(total_cnt,swarm.size,trap_list))

    '''plot the odor concentration field'''
    # conc_array = importedConc.array_at_time(t)
    # # image.set_data(scipy.log(conc_array))
    # image.set_data(conc_array)

    return swarm

frames = int(
frame_rate*(simulation_time)/times_real_time)

#Begin loop
for iteration in range(repeat_count):
    output_file = file_name+'_'+str(iteration+1)+'.pkl'
    #Re-initialize swarm and t each loop
    swarm = swarm_models.BasicSwarmOfFlies(importedWind,traps,param=swarm_param,
        start_type='fh',track_plume_bouts=False,track_arena_exits=False)
    t = 0.0
    for i in range(frames):
        update(i)
    # anim = FuncAnimation(fig, update, frames=frames,
    # init_func=init,repeat=False)
    # plt.show()
    # saved = anim.save(file_name+'.mp4', dpi=100, fps=frame_rate, extra_args=['-vcodec', 'libx264'])

    with open(output_file, 'w') as f:
        pickle.dump(swarm,f)


#end loop
