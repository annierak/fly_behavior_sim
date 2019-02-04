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
from pompy import data_importers

descriptor = '_pure_advection'
repeats = range(10)

file_name = '10_26_fly_sim'+descriptor

#Begin loop
for iteration in repeats:

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
    wind_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
    plume_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]

    importedWind = data_importers.ImportedWind(wind_file,release_delay)

    array_z = 0.01
    array_dim_x = 1000
    array_dim_y = array_dim_x
    puff_mol_amount = 1.

    #Now we're using the O(N+M) conc approx
    importedPlumes = data_importers.ImportedPlumes(plume_file,
        array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay,
        box_approx=True,epsilon = 0.0001)

    #Setup fly swarm
    swarm_size=2000

    wind_slippage = (0.0,0.0)
    flight_speed = 1.5
    odor_threshold = 0.05
    cast_delay= 3
    cast_interval = [1,3]
    cast_timeout = 20
    pure_advection = True

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
    release_times_hist = scipy.zeros(total_count)
    counter = 0
    for release_time,count in list(zip(times,flattened_release_counts)):
        if count>0:
            release_times_hist[counter:counter+count] = release_time
            counter +=count

    release_times = utility.draw_from_inputted_distribution(
        release_times_hist,2,swarm_size)


    heading_data = {'angles':(scipy.pi/180)*scipy.array([0.,90.,180.,270.]),
                    'counts':scipy.array([[1724,514,1905,4666],[55,72,194,192]])
                    }

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
            'cast_timeout':cast_timeout,
            'pure_advection':pure_advection
            }
    swarm = swarm_models.BasicSwarmOfFlies(importedWind,traps,param=swarm_param,
        start_type='fh',track_plume_bouts=False,track_arena_exits=False)

    frames = int(
    frame_rate*(simulation_time)/times_real_time)

    output_file = file_name+'_'+str(iteration+1)+'.pkl'
    #Re-initialize swarm and t each loop
    t = 0.0
    for i in range(frames):
        print('i='+str(i))
        # global swarm, t, release_delay,importedWind,importedPlumes,\
        # traps#,fly_dots,title,image
        for k in range(capture_interval):
            #update flies
            print('t: {0:1.2f}'.format(t))

            swarm.update(t,dt,importedWind,importedPlumes,traps,pre_stored=True) #for presaved plumes

            t+= dt
            time.sleep(0.001)

    with open(output_file, 'w') as f:
        pickle.dump((swarm,sys.argv[1]),f)

    #redraw the release times -- this is our source of randomness between simulations,
    #ALONG WITH the drawing from the heading distribution
    # release_times = utility.draw_from_inputted_distribution(
    #     release_times_hist,2,swarm_size)
    # swarm_param.update({'release_time':release_times})
    # importedWind = data_importers.ImportedWind(wind_file,release_delay)
    # importedPlumes = data_importers.ImportedPlumes(plume_file,
    #     array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay,
    #     box_approx=True,epsilon = 0.0001)


#end loop
