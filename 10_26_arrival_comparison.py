import scipy
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import odor_tracking_sim.utility as utility
import matplotlib.transforms



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
input_file = f+'.pkl'
with open(input_file,'r') as f:
    # swarm = pickle.load(f)
    swarm = pickle.load(f)

num_bins = 120

trap_num_list = swarm.get_trap_nums()


peak_counts = scipy.zeros(8)
rasters = []

fig = plt.figure(figsize=(11, 11))

sim_reorder = [3,2,1,8,7,6,5,4]

for i in range(8):
    row = sim_reorder[i]-1#2*(i)%8
    col = 0#(2*(i)-(2*(i)%8))/8
    ax = plt.subplot2grid((8,2),(row,col))
    t_sim = swarm.get_time_trapped(i)
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
        # ax.set_xticks([])
    #This is the raster plot option:
    # r = ax.eventplot(t,colors=['green'])[0]
    # rasters.append(r)
    #Add on a plot for the flies that went straight into the trap
labels = ['N','NE','E','SE','S','SW','W','NW']
side_labels = [85,22,18,2,15,377,317,188]

for i in range(len(total_counts)):
    t_obs = emp_arrival_times[i,:]
    t_obs = t_obs[~scipy.isnan(t_obs)]
    t_obs = t_obs/60.
    row = i#(2*i+1)%8
    col = 1#((2*i+1)-(2*i+1)%8)/8
    ax = plt.subplot2grid((8,2),(row,col))
    ax.set_yticks([])
    (n, bins, patches) = ax.hist(t_obs,num_bins,range=(0,max(t_obs)),
    color='green')
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



plt.show()
