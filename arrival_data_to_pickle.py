

import sys
from release_data_tools import load_release_data
import scipy
import matplotlib.pyplot as plt
import sys, os
import json
import collections
import cPickle as pickle

def convert_dict_from_unicode(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert_dict_from_unicode, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert_dict_from_unicode, data))
    else:
        return data

filename = sys.argv[1]

with open(filename) as f:
    dictionary = json.load(f)

min_pre_release = 10

sec_post_release = 3000
dt = [3,3,3,3,2,3,3,3]

num_traps = 8
arrival_count_array = scipy.zeros((num_traps,sec_post_release))

trap_row = 0
for dt,trap_id in list(zip(dt,dictionary)):
    print trap_id
    trap_data = dictionary[trap_id] #track data is itself a dictionary
    time = trap_data['time_since_release']
    fly_arrivals = trap_data['flies_in_frame']
    # fly_arrivals_post_release = fly_arrivals[
    # int(60*min_pre_release*frame_rate):]
    arrival_count_array[trap_row,time] = fly_arrivals
    trap_row+=1

trap_order = [4,3,6,5,0,2,1,7]
arrival_count_array = arrival_count_array[trap_order,:]

# print(scipy.size(release_count_array,1))
plt.imshow(arrival_count_array,aspect=100)
# plt.show()

output_file = 'empirical_arrival_counts.pkl'
with open(output_file, 'w') as f:
    pickle.dump(arrival_count_array,f)
