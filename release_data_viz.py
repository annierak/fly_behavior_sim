import scipy
import matplotlib.pyplot as plt
import cPickle as pickle
import matplotlib.gridspec as gridspec

#construct release distribution
empirical_release_data_file = 'empirical_release_counts.pkl'
with open(empirical_release_data_file,'r') as f:
    release_data_counts = pickle.load(f)

#turn release counts into a list of times
# flattened_release_counts = scipy.sum(release_data_counts,0).astype(int)
emp_dt = 1./25
# times = scipy.linspace(dt,len(flattened_release_counts)*emp_dt,
#     len(flattened_release_counts))
# total_count = int(sum(flattened_release_counts))
# release_times = scipy.zeros(total_count)
# counter = 0
# for release_time,count in list(zip(times,flattened_release_counts)):
#     if count>0:
#         release_times[counter:counter+count] = release_time
#         counter +=count

times = scipy.linspace(emp_dt,scipy.shape(release_data_counts)[1]*emp_dt,
    scipy.shape(release_data_counts)[1])
total_counts = scipy.sum(release_data_counts,1).astype(int)
print(total_counts)
emp_release_times = scipy.full((len(total_counts),max(total_counts)),scipy.nan)

for trap_row in range(len(total_counts)):
    counter = 0
    for time,count in list(zip(
        times,release_data_counts[trap_row,:].astype(int))):
        if count>0:
            emp_release_times[trap_row,counter:counter+count] = time
            counter +=count

# plt.imshow(emp_release_times,aspect=500)
# plt.show()
rows = scipy.shape(emp_release_times)[0]

labels = ['N','N','E','E','S','S','W','W']

emp_release_times = emp_release_times/60.

bins = 100
peak = 0

gs1 = gridspec.GridSpec(4, 1)
gs1.update(wspace=0.025, hspace=0.05)
rows = [1,3,5,7]
plt.figure()

for i in range(4):
    ax = plt.subplot(4,1,i+1)
    (n,bins,patches) = plt.hist(emp_release_times[rows[i],~scipy.isnan(
        emp_release_times[rows[i],:])],bins=bins)
    plt.ylabel(labels[rows[i]],rotation=0,size=20)
    peak = max(peak,max(n))
    if i==0:
        ax.set_title('Departure Statistics')
    if i==3:
        plt.xlabel('time (min)')
    plt.yticks([])
    ax.yaxis.labelpad = 30

# for row in range(4):
#     plt.subplot(4,1,row+1)
#     plt.ylim([0,peak])
#     plt.yticks([0,peak])
plt.show()
