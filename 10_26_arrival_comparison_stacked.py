'''This script does what 10_26_arrival_comparison.py does in the case
where there are multiple simulations with the same parameters--stacks
their histograms into one. '''

import scipy
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import odor_tracking_sim.utility as utility
import matplotlib.transforms
import plotting_utls as pltus

empirical_color = 'green'
sim_color = 'blue'

#empirical arrival times
empirical_arrival_data_file = 'empirical_arrival_counts.pkl'
with open(empirical_arrival_data_file,'r') as f:
    arrival_data_counts = pickle.load(f)
times = scipy.linspace(1.,scipy.shape(arrival_data_counts)[1],
    scipy.shape(arrival_data_counts)[1])
total_counts = scipy.sum(arrival_data_counts,1).astype(int)
emp_arrival_times = scipy.full((len(total_counts),max(total_counts)),scipy.nan)

for trap_row in range(len(total_counts)):
    counter = 0
    for time,count in list(zip(
        times,arrival_data_counts[trap_row,:].astype(int))):
        if count>0:
            emp_arrival_times[trap_row,counter:counter+count] = time
            counter +=count


#the swarm (simulated) data
f = sys.argv[1]
num_trials = int(sys.argv[2])
input_files = [f+'_'+str(i+1)+'.pkl' for i in range(num_trials)]
swarms = []

# c = 1
for input_file in input_files:
        # if c==3:
        #     (swarm,plume_file_id) = pickle.load(f)
        # else:
    try:
        with open(input_file,'r') as f:
            (swarm,plume_file_id) = pickle.load(f)
    except(TypeError):
        # print(input_file)
        f.close()
        with open(input_file,'r') as f:
            swarm = pickle.load(f)

        # c+=1
    swarms.append(swarm)

num_bins = 120

trap_num_list = swarms[0].get_trap_nums()


peak_counts = scipy.zeros(8)
rasters = []

fig = plt.figure(figsize=(11, 11))

sim_reorder = scipy.array([3,2,1,8,7,6,5,4])

#Simulated histogram
for i in range(8):
    row = sim_reorder[i]-1
    col = 0
    ax = plt.subplot2grid((8,2),(row,col))
    t_sim = scipy.concatenate(tuple(swarm.get_time_trapped(i) for swarm in swarms))
    if len(t_sim)==0:
        ax.set_xticks([0,10,20,30,40,50])
        trap_total = 0
        pass
    else:
        t_sim = t_sim/60.
        (n, bins, patches) = ax.hist(t_sim,num_bins,range=(0,max(t_sim)))
        trap_total = int(sum(n))
        try:
            peak_counts[i]=max(n)
        except(IndexError):
            peak_counts[i]=0
    ax.set_xlim([0,50])
    if sim_reorder[i]-1==0:
         ax.set_title('Simulated')
    ax.set_yticks([])
    ax.text(-0.1,0.5,str(trap_total),transform=ax.transAxes,fontsize=20,horizontalalignment='center')
    if sim_reorder[i]-1==7:
        ax.set_xlabel('Time (min)',x=0.5,horizontalalignment='center')
    else:
        ax.set_xticklabels('')

plt.text(0.5,0.95,sys.argv[1],fontsize=15,transform=plt.gcf().transFigure,horizontalalignment='center')

labels = ['N','NE','E','SE','S','SW','W','NW']
side_labels = scipy.array([85,22,18,2,15,377,317,188])

#Observed histogram
for i in range(len(total_counts)):
    t_obs = emp_arrival_times[i,:]
    t_obs = t_obs[~scipy.isnan(t_obs)]
    t_obs = t_obs/60.
    row = i#(2*i+1)%8
    col = 1#((2*i+1)-(2*i+1)%8)/8
    ax = plt.subplot2grid((8,2),(row,col))
    ax.set_yticks([])
    (n, bins, patches) = ax.hist(t_obs,num_bins,range=(0,max(t_obs)),
    color=empirical_color)
    if i==7:
        ax.set_xlabel('Time (min)',x=0.5)
    else:
        # ax.set_xticks([])
        ax.set_xticklabels('')
    # ax.yaxis.labelpad = 30
    # ax.set_yticks([ax.get_ylim()[1]])
    ax.get_yaxis().set_tick_params(direction='out')
    # ax.set_yticks([9])
    # Create offset transform by 5 points in y direction
    dx = 0/72.; dy = -5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for label in ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax.text(-0.1,0.5,str(labels[i]),transform=ax.transAxes,fontsize=20,horizontalalignment='center')
    ax.text(1.1,0.5,str(side_labels[i]),transform=ax.transAxes,fontsize=20,horizontalalignment='center')

    if i==0:
         ax.set_title('Empirical')

fig2 = plt.figure(figsize=(10,6))

#Trap loc simulated histogram
trap_locs = (2*scipy.pi/swarm.num_traps)*scipy.array(swarm.list_all_traps())
sim_trap_counts = scipy.zeros(swarm.num_traps)

for swarm in swarms:
    sim_trap_counts += swarm.get_trap_counts()

# Way 0: polar bar plot
# ax = plt.subplot(1,2,1, polar=True)
# ax.bar(trap_locs,sim_trap_counts,align='center')
# ax.set_yticklabels('')
# plt.title('Simulated')
#
# ax = plt.subplot(1,2,2, polar=True)
# plt.bar(trap_locs,side_labels[sim_reorder-1],align='center')
# ax.set_yticklabels('')
# plt.title('Empirical')
# plt.show()

# First, the way with two circle plots
radius_scale = 0.3
plot_size = 1.3

ax = plt.subplot(1,2,1,aspect=1)
trap_locs_2d = [(scipy.cos(trap_loc),scipy.sin(trap_loc)) for trap_loc in trap_locs]
patches = [plt.Circle(center, size) for center, size in zip(trap_locs_2d, radius_scale*sim_trap_counts/max(sim_trap_counts))]
coll = matplotlib.collections.PatchCollection(patches, facecolors=sim_color,edgecolors=sim_color)
ax.add_collection(coll)
ax.set_ylim([-plot_size,plot_size]);ax.set_xlim([-plot_size,plot_size])
pltus.strip_bare(ax)
ax.text(0,1.15,'N',horizontalalignment='center',fontsize=10)
plt.title('Simulated')

ax = plt.subplot(1,2,2, aspect=1)
patches = [plt.Circle(center, size) for center, size in zip(trap_locs_2d, radius_scale*side_labels[sim_reorder-1]/max(side_labels))]
coll = matplotlib.collections.PatchCollection(patches, facecolors=empirical_color,edgecolors=empirical_color)
ax.add_collection(coll)
ax.set_ylim([-plot_size,plot_size]);ax.set_xlim([-plot_size,plot_size])
pltus.strip_bare(ax)
ax.text(0,1.15,'N',horizontalalignment='center',fontsize=10)
plt.title('Empirical')

plt.text(0.5,0.9,sys.argv[1],fontsize=15,transform=plt.gcf().transFigure,horizontalalignment='center')


plt.show()
