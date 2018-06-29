import scipy
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import odor_tracking_sim.utility as utility

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

num_bins = 120


labels = ['N','NE','E','SE','S','SW','W','NW']

for i in range(len(total_counts)):
    t_obs = emp_arrival_times[i,:]
    t_obs = t_obs[~scipy.isnan(t_obs)]
    t_obs = t_obs/60.
    ax = plt.subplot(8,1,i+1)
    (n, bins, patches) = ax.hist(t_obs,num_bins,range=(0,max(t_obs)),
    color='green')
    if i==7:
        ax.set_xlabel('Time (min)',x=0.5)
    else:
        # ax.set_xticks([])
        ax.set_xticklabels('')
    ax.set_ylabel(labels[i],rotation=0,size=20)
    ax.yaxis.labelpad = 30
    ax.set_yticks([])
    if i==0:
         ax.set_title('Arrival Statistics ')


plt.show()
