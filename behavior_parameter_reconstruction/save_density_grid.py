#Given inputted behavioral parameters, run a simulation and save a hdf5 file
#with the fly density over space per unit time

import time
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
matplotlib.use("Agg")
import sys
import itertools
import json
import cPickle as pickle
from matplotlib.animation import FuncAnimation
import datetime


import h5_logger
import h5py
import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
#This is hacky -- should turn data_importers into a module
sys.path.insert(0, '../')
import data_importers

file_name = '10_26_fly_sim'
output_file = file_name+'.pkl'

dt = 0.25
frame_rate = 20
times_real_time = 5 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))

simulation_time = 10*60. #seconds
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
wind_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
plume_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]

try:
    conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[3]
    importedConc = data_importers.ImportedConc(conc_file,release_delay)
    plotting=True
except(IndexError):
    plotting=False

plotting=True


importedWind = data_importers.ImportedWind(wind_file,release_delay)
wind_params = importedWind.run_param

array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.

importedPlumes = data_importers.ImportedPlumes(plume_file,
    array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay,box_approx=True)
plume_params = importedPlumes.run_param

#Setup fly swarm
wind_slippage = (0.,0.)
swarm_size=1000
use_empirical_release_data = False



#construct release distribution
if use_empirical_release_data:
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
else:

    #Grab wind info to determine heading mean
    wind_x,wind_y = importedWind.quiver_at_time(0)
    heading_mean = scipy.arctan(wind_y[0,0]/wind_x[0,0])

    beta = 10.
    release_times = scipy.random.exponential(beta,(swarm_size,))
    kappa = 2.

    heading_data=None

swarm_param = {
        'swarm_size'          : swarm_size,
        'heading_data'        : heading_data,
        'initial_heading'     : scipy.random.vonmises(heading_mean,kappa,(swarm_size,)),
        'x_start_position'    : scipy.zeros(swarm_size),
        'y_start_position'    : scipy.zeros(swarm_size),
        'flight_speed'        : scipy.full((swarm_size,), 0.5),
        'release_time'        : release_times,
        'release_delay'       : 0.,
        'cast_interval'       : [1, 3],
        'wind_slippage'       : wind_slippage,
        'odor_thresholds'     : {
            'lower': 0.0005,
            'upper': 0.001
            },
        'schmitt_trigger':False,
        'low_pass_filter_length':3, #seconds
        'dt_plot': capture_interval*dt,
        't_stop':simulation_time
        }

swarm = swarm_models.BasicSwarmOfFlies(importedWind,traps,param=swarm_param,
    start_type='fh',track_plume_bouts=False,track_arena_exits=False)

if plotting:
    # Set up figure

    im_extents = importedPlumes.sim_region
    xmin,ymin,xmax,ymax = im_extents.as_tuple()
    plt.ion()
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111)

    #Initial concentration plotting
    try:
        vmin,vmax,cmap = importedConc.get_image_params()
        image = importedConc.plot(0)
    except(NameError):
        pass
    buffr = -300
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

    #Initial wind plotting -- subsampled
    u,v = importedWind.quiver_at_time(0)
    full_size = scipy.shape(u)[0]
    shrink_factor = 10
    u,v = u[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor],\
        v[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor]
    x_origins,y_origins = importedWind.get_plotting_points()
    x_origins,y_origins = x_origins[0:-1:full_size-1],\
        y_origins[0:-1:full_size-1]
    x_origins,y_origins = x_origins[0:full_size-1:shrink_factor],\
        y_origins[0:full_size-1:shrink_factor]
    coords = scipy.array(list(itertools.product(x_origins, y_origins)))
    x_coords,y_coords = coords[:,0],coords[:,1]
    vector_field = ax.quiver(x_coords,y_coords,u,v)
    # print(scipy.unique(x_origins))
    vector_field = ax.quiver(x_coords,y_coords,u,v)
    #title with trapped count
    trap_list = []
    for trap_num, trap_loc in enumerate(traps.param['source_locations']):
        mask_trap = swarm.trap_num == trap_num
        trap_cnt = mask_trap.sum()
        trap_list.append(trap_cnt)
    total_cnt = sum(trap_list)
    title = plt.title('{0}/{1}:{2}'.format(total_cnt,swarm.size,trap_list))

    # frames = int(frame_rate*simulation_time/times_real_time)+1

    #Setup video
    FFMpegWriter = animate.writers['ffmpeg']
    metadata = {'title':file_name,}
    writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
    writer.setup(fig, file_name+'.mp4', 500)

# Setup density logger
density_logger_bin_width = 30.
density_logger_bin_edges = scipy.arange(importedPlumes.sim_region.x_min,
    importedPlumes.sim_region.x_max+1,density_logger_bin_width)
n = datetime.datetime.utcnow()
logger_filename = 'fly_density_timecourse_{0}.{1}-{2}:{3}_'.format(
    n.month,n.day,n.hour,n.minute)+file_name+'.hdf5'
density_logger_param = {}
# Density logger metadata requires:
# (1) swarm params
density_logger_param.update(swarm.param)
# (2) Plume and wind params
density_logger_param.update(importedWind.run_param)
density_logger_param.update(importedPlumes.run_param)
# (3) Data duration, timestep, and xy bin size
density_logger_param.update({'sim_duration': simulation_time,
    'sim_dt': dt , 'xy density bin width': density_logger_bin_width,
    'bin_edges':density_logger_bin_edges })

#The density log metadata, which comprises the swarm metadata, wind and plume metadata,
#needs to be compatible with json serialization--turn arrays to lists and
#just keep the name part of the rv_frozen objects (**** could store more info
#on them if necessary)

for key,value in density_logger_param.iteritems():
    if isinstance(value,scipy.stats._distn_infrastructure.rv_frozen):
        density_logger_param[key] = value.dist.name
    if isinstance(value,np.ndarray):
        density_logger_param[key] = value.tolist()
        # print(key,np.shape(value))



timecourse_logger = h5_logger.H5Logger(logger_filename,param_attr=density_logger_param)
#
#
plt.figure(10)
imd = plt.imshow(np.random.randn(1000,1000))



t = 0.0

while t<simulation_time:
    for k in range(capture_interval):
        #update flies
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        # swarm.update(t,dt,importedWind,importedConc,traps,pre_stored=True) #for presaved conc
        swarm.update(t,dt,importedWind,importedPlumes,traps,pre_stored=True) #for presaved plumes
        t+= dt
        time.sleep(0.001)

        #Save density to logger
        grid_hist,_,_ = scipy.histogram2d(         # Compute location bin membership using 2d hist function
            swarm.x_position,swarm.y_position,bins=
            [density_logger_bin_edges,density_logger_bin_edges])
        imd.set_data(grid_hist)
        data = {'grid_hist': grid_hist}
        timecourse_logger.add(data)


    # Update live display
    '''plot the flies'''
    #plot the wind vector field
    if plotting:
        u,v = importedWind.quiver_at_time(t)
        u,v = u[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor],\
            v[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor]

        fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

        fly_colors = [color_dict[mode] for mode in swarm.mode]
        fly_dots.set_color(fly_colors)

        trap_list = []
        for trap_num, trap_loc in enumerate(traps.param['source_locations']):
            mask_trap = swarm.trap_num == trap_num
            trap_cnt = mask_trap.sum()
            trap_list.append(trap_cnt)
        total_cnt = sum(trap_list)
        title.set_text('{0}/{1}:{2}'.format(total_cnt,swarm.size,trap_list))

        '''plot the odor concentration field'''
        try:
            conc_array = importedConc.array_at_time(t)
            image.set_data(scipy.log(conc_array))
        except(NameError):
            pass

        #Update time display
        release_delay = release_delay/60.
        if t<release_delay*60.:
            text ='-{0} min {1} sec'.format(int(scipy.floor(abs(t/60.-release_delay))),int(scipy.floor(abs(t-release_delay*60)%60.)))
        else:
            text ='{0} min {1} sec'.format(int(scipy.floor(t/60.-release_delay)),int(scipy.floor(t%60.)))
        timer.set_text(text)

        plt.pause(0.001)

        writer.grab_frame()
        fig.canvas.flush_events()



if plotting:
    writer.finish()

with open(output_file, 'w') as f:
    pickle.dump(swarm,f)
