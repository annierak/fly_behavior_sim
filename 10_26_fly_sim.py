import time
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import sys
import itertools
import h5py
import json
import cPickle as pickle
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np



import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
from pompy import data_importers

file_name = 'michael_video_nov_2018_3'
output_file = file_name+'.pkl'

dt = 0.25
frame_rate = 20
times_real_time = 60 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))


simulation_time = 30*60. #seconds
release_delay = 20.*60

wind_vector_field = False


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
# conc_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[1]
# wind_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[2]
# plume_file = '/home/annie/work/programming/pompy_duplicate/'+sys.argv[3]

conc_file = '/home/annie/work/programming/special_data/'+sys.argv[1]
wind_file = '/home/annie/work/programming/special_data/'+sys.argv[2]
plume_file = '/home/annie/work/programming/special_data/'+sys.argv[3]


importedConc = data_importers.ImportedConc(conc_file,release_delay)
importedWind = data_importers.ImportedWind(wind_file,release_delay)

array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.

importedPlumes = data_importers.ImportedPlumes(plume_file,
    array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay,
    box_approx=True,epsilon = 0.0001)



#Setup fly swarm
wind_slippage = (0.,0.)
swarm_size=1000

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
swarm_param = {
        'swarm_size'          : swarm_size,
        'heading_data'        : heading_data,
        'x_start_position'    : scipy.zeros(swarm_size),
        'y_start_position'    : scipy.zeros(swarm_size),
        'flight_speed'        : scipy.full((swarm_size,), 1.5),
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

vmin,vmax,cmap = importedConc.get_image_params()
im_extents = importedConc.simulation_region
xmin,xmax,ymin,ymax = im_extents
# Set up figure
# plt.ion()
fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111)

#Initial concentration plotting
image = importedConc.plot(0)
cmap = matplotlib.colors.ListedColormap(['white', 'orange'])
# image.set_cmap(plt.cm.get_cmap("BuGn"))
image.set_cmap(cmap)

buffr = -300
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
ax.text(1.,1.02,'time since release:',color='r',transform=ax.transAxes,
    horizontalalignment='right')


#Initial wind plotting -- subsampled
if wind_vector_field:
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
    vector_field = ax.quiver(x_coords,y_coords,u,v)
else:
    u,v = importedWind.quiver_at_time(0)
    wind_x,wind_y = u[0,0],v[0,0]
    arrow_size= .05
    arrow_loc = 0.9,0.9
    arrowstyle=matplotlib.patches.ArrowStyle.Fancy(head_length=2, head_width=2, tail_width=.4)
    wind_arrow = matplotlib.patches.FancyArrowPatch((arrow_loc[0], arrow_loc[1]),
    (arrow_loc[0]+arrow_size*wind_x, arrow_loc[1]+arrow_size*wind_y),
    transform=ax.transAxes,color='b',mutation_scale=10,arrowstyle=arrowstyle)
    ax.add_patch(wind_arrow)
    ax.text(0.9,0.92,'Wind',
        transform=ax.transAxes,color='b',horizontalalignment='center')


#traps
for x,y in traps.param['source_locations']:

    #Black x
    plt.scatter(x,y,marker='x',s=50,c='k')

    #Red circles
    # p = matplotlib.patches.Circle((x, y), 15,color='red')
    # ax.add_patch(p)



#title with trapped count
trap_list = []
for trap_num, trap_loc in enumerate(traps.param['source_locations']):
    mask_trap = swarm.trap_num == trap_num
    trap_cnt = mask_trap.sum()
    trap_list.append(trap_cnt)
total_cnt = sum(trap_list)
# title = plt.title('{0}/{1}:{2}'.format(total_cnt,swarm.size,trap_list))


#Remove plot edges and add scale bar
fig.patch.set_facecolor('white')
#
# scalebar = AnchoredSizeBar(ax.transData,
#                        100, '100 m', 'upper left',
#                        pad=0.1,
#                        frameon=False,
#                        size_vertical=1)
# ax.add_artist(scalebar)
plt.plot([-900,-800],[900,900],transform=ax.transData,color='k')
ax.text(-900,820,'100 m')
plt.axis('off')

#Fly behavior color legend
for mode,fly_facecolor,fly_edgecolor,a in zip(
    ['Dispersing','Surging','Casting','Trapped'],
    facecolor_dict.values(),
    edgecolor_dict.values(),
    [0,50,100,150]):

    plt.scatter([1000],[-600-a],edgecolor=fly_edgecolor,
    facecolor=fly_facecolor, s=20)
    plt.text(1050,-600-a,mode,verticalalignment='center')



# plt.show()

FFMpegWriter = animate.writers['ffmpeg']
metadata = {'title':file_name,}
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, file_name+'.mp4', 500)
plt.ion()


t = 0.0 #- release_delay

while t<simulation_time:
    for k in range(capture_interval):
        #update flies
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        # swarm.update(t,dt,importedWind,importedConc,traps,pre_stored=True) #for presaved conc
        swarm.update(t,dt,importedWind,importedPlumes,traps,pre_stored=True) #for presaved plumes
        #Update time display
        release_delay = release_delay/60.
        if t<release_delay*60.:
            text ='-{0} min {1} sec'.format(int(scipy.floor(abs(t/60.-release_delay))),int(scipy.floor(abs(t-release_delay*60)%60.)))
        else:
            text ='{0} min {1} sec'.format(int(scipy.floor(t/60.-release_delay)),int(scipy.floor(t%60.)))
        timer.set_text(text)
        t+= dt
        time.sleep(0.001)
    # Update live display
    '''plot the flies'''
    #plot the wind vector field
    u,v = importedWind.quiver_at_time(t)
    if wind_vector_field:
        u,v = u[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor],\
            v[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor]
        vector_field.set_UVC(u,v)
    else:
        wind_x,wind_y = u[0,0],v[0,0]
        wind_arrow.set_positions(
        (arrow_loc[0], arrow_loc[1]),
            (arrow_loc[0]+arrow_size*wind_x, arrow_loc[1]+arrow_size*wind_y))



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
    # title.set_text('{0}/{1}:{2}'.format(total_cnt,swarm.size,trap_list))

    '''plot the odor concentration field'''
    conc_array = importedConc.array_at_time(t)

    # non_inf_log =
    log_im = scipy.log(conc_array)
    cutoff_l = scipy.percentile(log_im[~scipy.isinf(log_im)],10)
    cutoff_u = scipy.percentile(log_im[~scipy.isinf(log_im)],99)

    # im = (log_im>cutoff_l) & (log_im<0.1)
    # n = matplotlib.colors.Normalize(vmin=0,vmax=1)
    # image.set_data(im)
    # image.set_norm(n)

    image.set_data(log_im)
    n = matplotlib.colors.Normalize(vmin=cutoff_l,vmax=cutoff_u)
    image.set_norm(n)



    # image.set_data(conc_array)


    plt.pause(0.001)

    writer.grab_frame()
    fig.canvas.flush_events()


writer.finish()

with open(output_file, 'w') as f:
    pickle.dump(swarm,f)


    # anim = FuncAnimation(fig, update, frames=frames, repeat=False,save_count = frames)
    # saved = anim.save(file_name+'.mp4', dpi=100, fps=frame_rate,
    # extra_args=['-vcodec', 'libx264'])
