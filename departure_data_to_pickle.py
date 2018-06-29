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

num_traps = len(dictionary)
min_pre_release = 10
min_post_release = 60

frame_rate = 25.
dt = 1/frame_rate


release_count_array = scipy.zeros((num_traps,int(
60*min_post_release*frame_rate)))

trap_row = 0
for trap_id in dictionary:
    print(trap_id)
    track_data = dictionary[trap_id] #track data is itself a dictionary
    track_data = convert_dict_from_unicode(track_data)
    time = track_data['time_seconds']
    fly_arrivals = track_data['flies_in_frame']
    fly_arrivals_post_release = fly_arrivals[
    int(60*min_pre_release*frame_rate):]
    if trap_row==0:
        release_count_array = release_count_array[:,0:len(fly_arrivals_post_release)]
    release_count_array[trap_row,:] = fly_arrivals_post_release
    trap_row+=1

# print(scipy.size(release_count_array,1))
# plt.imshow(release_count_array,aspect=1000)
# plt.show()

output_file = 'empirical_release_counts.pkl'
with open(output_file, 'w') as f:
    pickle.dump(release_count_array,f)
