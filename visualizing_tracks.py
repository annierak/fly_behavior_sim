import sys
from release_data_tools import load_release_data
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import json
import collections

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

fig = plt.figure()
minutes_of_video_pre_release = 10
minutes_of_video_post_release = 50
total_min = minutes_of_video_pre_release +minutes_of_video_post_release
frame_rate = 25
counter = 1
subplot_num = len(dictionary)

for trap_id in dictionary:
    print trap_id
    track_data = dictionary[trap_id] #track data is itself a dictionary
    track_data = convert_dict_from_unicode(track_data)
    # time = np.linspace(0,total_min*60*1.1, total_min*60*frame_rate*1.1, endpoint = False)
    # flies_in_frame = np.zeros(total_min*60*frame_rate*1.1)

    time = track_data['time_seconds']
    flies_in_frame = track_data['flies_in_frame']

    ax = fig.add_subplot(subplot_num, 1, counter)
    ax.plot(time,flies_in_frame, 'k')
    ax.set_title(trap_id)
    ax.set_ylim([-0.5, max(flies_in_frame[60*frame_rate*minutes_of_video_pre_release:-1])+1]) #
    ax.axvline(minutes_of_video_pre_release*60)
    if counter != subplot_num:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('time, seconds')
        ax.set_ylabel('tracked flies')
    counter +=1

plt.show()
